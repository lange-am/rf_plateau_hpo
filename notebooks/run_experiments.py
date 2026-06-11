"""
run_experiments.py
------------------

Experiment orchestration helpers for the Random Forest HPO study specified
in the project-level CITATION.cff file.

The module wraps the public tuning routines from ``rf_plateau_hpo.core`` and
constructs the experimental protocols used in the paper:

- ``TPE``: classic Optuna/TPE baseline with a fixed ``n_estimators`` range.
- ``HB``: Hyperband-style multi-fidelity baseline using ``n_estimators`` as the resource.
- ``ES``: naive monotone early-stopping baseline based on the left plateau condition.
- ``PLATEAU``: proposed triplet-based plateau-search algorithm.
- ``TPE_Tmin`` / ``TPE_Tmin-Tmax`` / ``TPE_Tmin-ES`` / ``TPE_Tmin-PLT``:
  two-stage decoupled variants used for ablation studies.

All experiments are logged and saved as ``.dill`` files. Study metadata includes
best-trial summaries, selected tree counts, pruning status, tree-building cost,
and, for PLATEAU runs, trial-wise triplet trajectories. The module also provides
helpers for building parameter grids and for running large experiment queues via
``cpu_pinning.run_queue_pinned``.

Key functions
-------------
- ``run_experiment()``: run a single tuning experiment with a selected method.
- ``process_dataset()``: generate a full set of configurations for one dataset
  and execute them with CPU pinning and optional background file moving.
- ``parse_study()`` / ``parse_log_tail()``: extract metrics from Optuna studies
  and log files for downstream analysis.

Copyright (c) 2025-2026 Andrey Lange and rf_plateau_hpo contributors.
Licensed under the MIT License. See the LICENSE file in the project root for details.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
from inspect import signature
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

try:
    from typing import Literal, get_args
except ImportError:  # pragma: no cover
    from typing_extensions import Literal, get_args

import dill
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import seaborn as sns

sns.set(style="whitegrid")

# --- Local / application ---
from cpu_pinning import run_queue_pinned
from split_common_params import split_common_params
from merge_safe import merge_safe
from file_mover import FileMoverThread, clean_source_dir

from rf_plateau_hpo.core import (
    DEFAULT_MAX_FEATURES_GRID,
    DEFAULT_MAX_DEPTH_RANGE,
    DEFAULT_MIN_SAMPLES_LEAF_RANGE,
    DEFAULT_MIN_SAMPLES_SPLIT_RANGE,
    DEFAULT_T_MAX,
    DEFAULT_N_ESTIMATORS_LADDER,
    DEFAULT_HYPERBAND_REDUCTION_FACTOR,
    DEFAULT_N_ESTIMATORS_START,
    DEFAULT_MAX_TREES,
    tune_rf_oob,
    tune_rf_oob_bohb,
    tune_rf_oob_plateau,
)

RF_HPO_ALGORITHMS = Literal[
    "TPE_Tmin",
    "TPE_Tmin-Tmax",
    "TPE_Tmin-ES",
    "TPE_Tmin-PLT",
    "TPE",
    "HB",
    "ES",
    "PLATEAU",
]
EXPERIMENT_SORT_RULE = Literal["random_state", "plateau_first"]

DEFAULT_METHOD_GRID = get_args(RF_HPO_ALGORITHMS)
DEFAULT_N_TRIALS_GRID = (40, 120)
DEFAULT_DELTA_GRID = (1e-3, 3e-3, 5e-3, 7e-3, 9e-3)
DEFAULT_SCALE_FACTOR_GRID = (1.5, 1.25, 1.75, 2.0)
DEFAULT_N_EXPERIMENTS = 20

# ---------- HELPER FUNCTIONS ----------

def _filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by `func`."""
    sig = signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def parse_study(study: optuna.Study) -> Dict[str, Any]:
    """
    Summarize an Optuna Study primarily using user_attrs produced by our objectives.
    Additionally, include BEST_* fields (trial number, score, params) from `study.best_trial`.
    """
    trials = list(study.trials)
    n_trials = len(trials)

    out = Counter()
    out["n_trials_total"] = n_trials

    trees_built = [0]*n_trials
    B           = [None]*n_trials
    pruned      = [False]*n_trials
    shift_left  = [False]*n_trials
    stay        = [False]*n_trials
    shift_right = [False]*n_trials

    n_trials_with_trees_built = 0
    for i, t in enumerate(trials):
        ua = t.user_attrs

        if "trees_built" in ua:
            out["n_trees_built"] += int(ua["trees_built"])
            n_trials_with_trees_built += 1
            trees_built[i] = int(ua["trees_built"])

        if "triplet" in ua:
            B[i] = ua["triplet"][1]

        if "pruned" in ua:
            out["n_trials_pruned_"+str(ua["pruned"])] += 1
            out["n_trials_pruned"] += 1
            pruned[i] = True

        if "shift" in ua:
            sh = int(ua["shift"])
            if sh == -1:
                out["n_trials_shift_left"] += 1
                shift_left[i] = True
            if sh == 0:
                out["n_trials_stay"] += 1
                stay[i] = True
            if sh == 1:
                out["n_trials_shift_right"] += 1
                shift_right[i] = True
                if "pruned" in ua:
                    out["n_trials_shift_right_pruned_"+str(ua["pruned"])] += 1

    out["trees_built"] = trees_built
    out["B"] = B
    out["pruned"] = pruned
    out["shift_left"] = shift_left
    out["stay"] = stay
    out["shift_right"] = shift_right

    if n_trials_with_trees_built != len(trials):
        warnings.warn(
            f"trees_built missing for some trials: have={n_trials_with_trees_built}, total={len(trials)}.",
            RuntimeWarning,
        )

    # --- BEST fields (top-level, log-style) ---
    try:
        bt = study.best_trial
        out["BEST_trial"] = bt.number
        out["BEST_score"] = bt.value
        out["BEST_params"] = dict(bt.params)
    except Exception:
        pass

    # --- Consistency checks using Optuna TrialState (warnings only) ---
    n_pruned_attr = out.get("n_trials_pruned", 0)
    n_pruned_state = sum(t.state == optuna.trial.TrialState.PRUNED for t in trials)
    if n_pruned_attr != n_pruned_state:
        warnings.warn(
            f"Pruned trials mismatch: user_attrs pruned={n_pruned_attr}, "
            f"TrialState.PRUNED={n_pruned_state}.",
            RuntimeWarning,
        )

    n_fail = sum(
        t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        for t in trials
    )
    if n_fail != 0:
        warnings.warn(f"Study contains non-(COMPLETE/PRUNED) trials: n_fail={n_fail}.", RuntimeWarning)

    return dict(out)


# parse the end-block of the log-file and extract `BEST'-parameters
def parse_log_tail(log_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse the final contiguous BEST-block from a log file.

    Expected segment format (in the 3rd '|' column and beyond):
        BEST_<key>=<python-literal-or-string>

    Returns a dict with keys BEST_<key> and, if available, 'Total time'.
    """
    p = Path(log_file)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}

    time_re = re.compile(r"\[t\+(\d+(?:\.\d+)?)s\]\s*$")

    def is_best_line(ln: str) -> bool:
        cols = [c.strip() for c in ln.split("|")]
        return len(cols) >= 3 and cols[2].startswith("BEST_")

    # Collect the last contiguous BEST-block
    block = []
    for ln in reversed(lines):
        if is_best_line(ln):
            block.append(ln)
        elif block:
            break
    block.reverse()
    if not block:
        return {}

    out: Dict[str, Any] = {}

    for ln in block:
        ln = time_re.sub("", ln)
        cols = [c.strip() for c in ln.split("|")]

        for seg in cols[2:]:
            seg = seg.strip()
            if not seg.startswith("BEST_"):
                continue
            body = seg[5:]  # after "BEST_"
            if "=" not in body:
                continue

            k, v = body.split("=", 1)
            try:
                out[f"BEST_{k.strip()}"] = ast.literal_eval(v.strip())
            except Exception:
                out[f"BEST_{k.strip()}"] = v.strip()

    m = time_re.search(block[-1])
    if m:
        out["time_total"] = float(m.group(1))

    return out


def sumup_common(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two parsed-study dicts.

    Rule:
      - For keys starting with 'n_' OR containing 'time' (substring), sum numeric values.
    """
    out = {}
    for k, v2 in d2.items():
        if (k.startswith("n_") or ("time" in k)) and (k in d1):
            v1 = d1[k]
            # assume numeric; let it raise if not (fail-fast)
            out[k] = v1 + v2

    return out


# ---------- CORE EXPERIMENT RUNNER ----------

def run_experiment(
    method: RF_HPO_ALGORITHMS,
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    # --- Search space ---
    max_features_grid: Sequence[object] = DEFAULT_MAX_FEATURES_GRID,
    max_depth_range: Tuple[int, int] = DEFAULT_MAX_DEPTH_RANGE,
    n_estimators_range: Tuple[int, int] = (DEFAULT_N_ESTIMATORS_START, DEFAULT_T_MAX),
    min_samples_leaf_range: Tuple[int, int] = DEFAULT_MIN_SAMPLES_LEAF_RANGE,
    min_samples_split_range: Tuple[int, int] = DEFAULT_MIN_SAMPLES_SPLIT_RANGE,
    tune_criterion: bool = True,
    # --- Pass to Random Forest class ---
    criterion: Optional[str] = None,
    class_weight: Optional[Union[str, Dict[str, float], List[Dict[str, float]]]] = None,
    # --- Hyperband-like resource ladder (n_estimators budgets) ---
    n_estimators_ladder: Sequence[int] = DEFAULT_N_ESTIMATORS_LADDER,
    hyperband_reduction_factor: int = DEFAULT_HYPERBAND_REDUCTION_FACTOR,
    hyperband_bootstrap_count: int = 0,
    hyperband_warmup_rungs: int = 0,
    # --- n_estimators triplet mechanics ---
    n_estimators_start: int = DEFAULT_N_ESTIMATORS_START,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    delta: float = DEFAULT_DELTA_GRID[0],
    max_trees: int = DEFAULT_MAX_TREES,
    # --- Optuna / runtime ---
    n_trials: int = DEFAULT_N_TRIALS_GRID[0],
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: int = 0,
    # --- Path and tags ---
    outdir: Union[str, Path] = "",
    dataset: str = "",
    note: str = "",
    # --- Resume / skip already completed runs ---
    files_done: Optional[Sequence[Union[str, Path]]] = None,
): # -> Tuple[Dict[str, Any], Path]:
    """
    Run experiment with specified tuning method and save results.
    
    """
    # Setup paths
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    random_state_ = (time.time_ns() ^ (os.getpid() << 16)) & 0x7FFFFFFF if random_state is None else random_state
    random_state_str = f"{random_state_:010}"

    def _output_file_name(
        method: RF_HPO_ALGORITHMS = method, n_trials: int = n_trials, ext: str = 'log.tmp'
    )-> Path:
        run_id = '-'.join([method, dataset, str(n_trials), random_state_str])
        return outdir / f"{run_id}.{ext}"    

    log_file  = _output_file_name()
    dill_file = _output_file_name(ext='dill.tmp')

    if files_done and dill_file in files_done:
        return

    # Prepare all parameters
    params = {
        'X': X, 'y': y, 
        'problem': problem, 
        'score_func': score_func,
        'greater_is_better': greater_is_better, 
        #
        'max_features_grid': max_features_grid,
        'max_depth_range': max_depth_range, 
        'n_estimators_range': n_estimators_range,
        'min_samples_leaf_range': min_samples_leaf_range,
        'min_samples_split_range': min_samples_split_range, 
        'tune_criterion': tune_criterion,
        #
        'criterion': criterion, 
        'class_weight': class_weight, 
        #
        'n_estimators_ladder': n_estimators_ladder,
        'hyperband_reduction_factor': hyperband_reduction_factor,
        'hyperband_bootstrap_count': hyperband_bootstrap_count,
        'hyperband_warmup_rungs': hyperband_warmup_rungs,
        #
        'n_estimators_start': n_estimators_start,
        'scale_factor': scale_factor,
        'delta': delta, 
        'max_trees': max_trees,
        #
        'n_trials': n_trials,
        'n_jobs': n_jobs, 
        'random_state': random_state_, 
        'verbose': verbose,
        #
        'log_file': log_file, 
        'refit': False,
    }

    classic_params  = _filter_kwargs(tune_rf_oob, params)
    bohb_params     = _filter_kwargs(tune_rf_oob_bohb, params)
    plateau_params  = _filter_kwargs(tune_rf_oob_plateau, params)

    def _parse(study: optuna.Study, log_file: Path) -> Dict[str, Any]:
        return {**parse_log_tail(log_file), **parse_study(study), 'study': study, 'log_file': log_file}

    study0, study = None, None
    log_file0 = None
    if "TPE_Tmin" in method:
        # Special baselines when n_estimators and other hyperparameters are tuned separately
        t_min, t_max = classic_params['n_estimators_range']

        log_file0 = _output_file_name(method="TPE_Tmin")
        _, study0 = tune_rf_oob(**{**classic_params, 'n_estimators_range': (t_min, t_min), 'log_file': log_file0})

        best_prm = study0.best_trial.params
        fixed_params = {
            'max_features_grid': (best_prm['max_features'],),
            'max_depth_range': (best_prm['max_depth'], best_prm['max_depth']),
            'min_samples_leaf_range': (best_prm['min_samples_leaf'], best_prm['min_samples_leaf']),
            'min_samples_split_range': (best_prm['min_samples_split'], best_prm['min_samples_split']),
            'tune_criterion': False,
            'criterion': best_prm.get('criterion', classic_params['criterion'])
        }

        if method == "TPE_Tmin":
            # TPE Tmin: classic HPO search with fixed T=t_min
            study0, study = None, study0
            log_file0, log_file = None, log_file0

        elif method == "TPE_Tmin-Tmax": 
            # TPE Tmin-Tmax: classic HPO search with fixed T=t_min + single evaluation at T=t_max
            log_file = _output_file_name(n_trials=1)
            _, study = tune_rf_oob(
                **{**classic_params, **fixed_params, 'n_estimators_range': (t_max, t_max), 'n_trials': 1, 'log_file': log_file}
            )

        elif method == "TPE_Tmin-ES":
            # TPE Tmin-ES: classic HPO search with fixed T=t_min + early stopping
            if n_estimators_ladder[0] != t_min:
                warnings.warn(
                    f"In '{method}' it is assumed that n_estimators_ladder[0]=Tmin."
                    f"(n_estimators_ladder[0]={n_estimators_ladder[0]}, Tmin={t_min})",
                    RuntimeWarning,
                )

            if hyperband_reduction_factor >= 2:
                warnings.warn(
                    f"In '{method}' it is assumed that hyperband_reduction_factor < 2 (no pruning)."
                    f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                    RuntimeWarning,
                )

            _, _, study, _ = tune_rf_oob_bohb(**{**bohb_params, **fixed_params})

        elif method == "TPE_Tmin-PLT":
            # TPE Tmin-PLT: classic HPO search with fixed T=t_min + plateau search 
            if n_estimators_start != t_min:
                warnings.warn(
                    f"In '{method}' it is assumed that n_estimators_start=Tmin."
                    f"(n_estimators_start={n_estimators_start}, Tmin={t_min})",
                    RuntimeWarning,
                )

            _, _, study, _ = tune_rf_oob_plateau(**{**plateau_params, **fixed_params})

    elif method == "TPE":
        # TPE: classic search over all hyperparameters
        _, study = tune_rf_oob(**classic_params)
    elif method == "ES":
        # ES: early-stopping with delta tolerance
        if hyperband_reduction_factor >= 2:
            warnings.warn(
                f"In '{method}' it is assumed that hyperband_reduction_factor < 2 (no pruning)."
                f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                RuntimeWarning,
            )

        _, _, study, _ = tune_rf_oob_bohb(**bohb_params)
    elif method == "HB":
        # HB: Hyperband-like with n_estimators budgets 
        if hyperband_reduction_factor < 2:
            warnings.warn(
                f"In '{method}' (Hyperband) it is assumed that hyperband_reduction_factor >= 2 (HyperbandPruner() is used)."
                f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                RuntimeWarning,
            )
        
        if delta >= 0.0:
            warnings.warn(
                f"In '{method}' (Hyperband) it is assumed that delta < 0 (no early stopping)."
                f"(delta={delta})",
                RuntimeWarning,
            )
        
        _, _, study, _ = tune_rf_oob_bohb(**bohb_params)
    elif method == "PLATEAU":
        # PLATEAU: plateau search
        _, _, study, _ = tune_rf_oob_plateau(**plateau_params)
    else:
        raise ValueError(f"Unknown method parameter: {method}")

    if study0 is not None:
        log_file0 = log_file0.rename(log_file0.with_suffix("")) # *.log.tmp -> *.log
        out0 = _parse(study0, log_file0)
    else:
        out0 = {}

    log_file = log_file.rename(log_file.with_suffix(""))
    out = _parse(study, log_file)

    output = sumup_common(out0, out)
    if out0:
        output.update(study0=out0)
    output.update(study=out)

    # Prepare dataset metadata
    data_params = {'dataset': dataset}
    if hasattr(X, 'shape'): data_params['X.shape'] = X.shape
    if hasattr(X, 'columns'): data_params['X.columns'] = X.columns
    if hasattr(y, 'name'): data_params['y.name'] = y.name
    
    # Remove large data objects before serialization
    del params['X'], params['y']
    
    run_params = {
        'experiment_started': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'method': method,
        'params_data': data_params,
        'params_in': params,
        'params_out': output,
        'note': note,
    }
    
    # Save to dill file
    with open(dill_file, "wb") as f:
        dill.dump(run_params, f, protocol=dill.HIGHEST_PROTOCOL)
    dill_file = dill_file.rename(dill_file.with_suffix(""))
    
    if verbose > 0:
        log_files = tuple(lf for lf in (log_file0, log_file) if lf is not None)
        print(f"Log files: {log_files}\nDill file: {dill_file}")
    
    # return run_params, dill_file


# ---------- CONFIGURATION GRID GENERATION ---------

def _method_needs_delta(method: RF_HPO_ALGORITHMS) -> bool:
    return method in ['TPE_Tmin-ES', 'ES', 'TPE_Tmin-PLT', 'PLATEAU']


def get_experiment_directory(
    dataset: Union[Path, str],
    tune_criterion: bool,
    depth_trees_only: bool,
    method: RF_HPO_ALGORITHMS,
    *,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    delta: float = DEFAULT_DELTA_GRID[0],
    n_trials: int = DEFAULT_N_TRIALS_GRID[-1],  
) -> Path:
    outdir = Path(dataset) / f"tune_criterion={tune_criterion}" / f"depth_trees_only={depth_trees_only}" / method
    outdir = outdir  / f"scale_factor={scale_factor}"
    if delta >= 0 and _method_needs_delta(method):
        outdir = outdir / f"delta={delta:.0e}".replace('e-0', 'e-')
    return outdir / f"n_trials={n_trials}"


def build_ladder(scale_factor: float, n_estimators_start: int, t_max: int):
    ladder = [n_estimators_start]
    ladder += [int(round(n_estimators_start * scale_factor))]
    
    while ladder[-1] < t_max:
        if ladder[-2] >= ladder[-1]:
            raise ValueError("n_estimators_ladder must be strictly increasing.")

        ladder.append(int(round(ladder[-1] * scale_factor)))
    return tuple(ladder)


def _get_run_experiment_configs(
    tune_criterion_grid: Sequence[bool] = (True, False),
    depth_trees_only_grid: Sequence[bool] = (True, False),
    method_grid: Sequence[RF_HPO_ALGORITHMS] = DEFAULT_METHOD_GRID,
    scale_factor_grid: Sequence[float] = DEFAULT_SCALE_FACTOR_GRID,
    delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
    n_trials_grid: Sequence[int] = DEFAULT_N_TRIALS_GRID,
    n_estimators_start: int = DEFAULT_N_ESTIMATORS_START,
    t_max: int = DEFAULT_T_MAX,
    max_trees: int = DEFAULT_MAX_TREES,
    n_experiments: int = DEFAULT_N_EXPERIMENTS,
    random_state: Optional[int] = None,
    dataset: str = "",
    **run_experiment_args,
) -> List[Dict[str, Any]]:
    configs = []
    for tune_criterion in tune_criterion_grid:
        for depth_trees_only in depth_trees_only_grid:
            for method in method_grid:
                for scale_factor in scale_factor_grid:

                    ladder = build_ladder(scale_factor, n_estimators_start, max_trees if method in ["TPE_Tmin-ES", "ES"] else t_max)
                    n_estimators_range = (n_estimators_start, ladder[-1])
                    # n_estimators_range = (100, 2565), ladder = (100, 150, 225, 338, 507, 760, 1140, 1710, 2565) 
                    # are expected when sf=1.5, n_estimators_start=100, t_max=2000

                    deltas = delta_grid if _method_needs_delta(method) else [-1.0]
                    for delta in deltas:
                        for n_trials in n_trials_grid:
                            outdir = get_experiment_directory(
                                dataset=dataset, 
                                tune_criterion=tune_criterion, 
                                depth_trees_only=depth_trees_only, 
                                method=method, 
                                scale_factor=scale_factor,
                                delta=delta, 
                                n_trials=n_trials,
                            )

                            cfg = dict(run_experiment_args)
                            cfg = merge_safe(cfg, {
                                'method': method,
                                'n_estimators_range': n_estimators_range,
                                'tune_criterion': tune_criterion,
                                'n_estimators_ladder': ladder,
                                'n_estimators_start': n_estimators_start,
                                'scale_factor': scale_factor,
                                'delta': delta,
                                'max_trees': max_trees,
                                'n_trials': n_trials,
                                'outdir': outdir,
                                'dataset': Path(dataset).name,
                            }, context="_get_run_experiment_configs")

                            if depth_trees_only:
                                cfg = merge_safe(cfg, {
                                    'min_samples_leaf_range': (1, 1),
                                    'min_samples_split_range': (2, 2),
                                    'max_features_grid': ('sqrt',),
                                }, context="_get_run_experiment_configs")
                                
                            if method in ['TPE_Tmin-ES', 'ES']:
                                cfg = merge_safe(
                                    cfg, {'hyperband_reduction_factor': 1}, 
                                    context="_get_run_experiment_configs"
                                )

                            for K in range(n_experiments):
                                cfg_ = dict(cfg)
                                cfg_ = merge_safe(cfg_, {
                                    'random_state': None if random_state is None else random_state+K,
                                    'note': str(K),
                                }, context="_get_run_experiment_configs")
                                configs += [cfg_]
    return configs


def _get_dataset_configs(
    tune_criterion_grid: Sequence[bool] = (True, False),
    depth_trees_only_grid: Sequence[bool] = (True, False),
    method_grid: Sequence[RF_HPO_ALGORITHMS] = DEFAULT_METHOD_GRID,
    scale_factor_grid: Sequence[float] = DEFAULT_SCALE_FACTOR_GRID,
    delta_grid: Sequence[float] = DEFAULT_DELTA_GRID,
    n_trials_grid: Sequence[int] = DEFAULT_N_TRIALS_GRID,
    **kwargs,
) -> List[Dict[str, Any]]:
    configs = []

    # "TPE" vs "PLATEAU"
    method_grid_ = tuple(meth for meth in method_grid if meth in ['TPE', 'PLATEAU'])
    if method_grid_:
        configs += _get_run_experiment_configs(
            tune_criterion_grid=tune_criterion_grid,
            depth_trees_only_grid=depth_trees_only_grid,
            method_grid=method_grid_,
            scale_factor_grid=(scale_factor_grid[0],),
            delta_grid=(delta_grid[0],),
            n_trials_grid=n_trials_grid,
            **kwargs,
        )
    
    # turn off (False) tune_criterion and depth_only
    tune_criterion_grid_ = (False,) if False in tune_criterion_grid else ()
    depth_trees_only_grid_ = (False,) if False in depth_trees_only_grid else ()

    # other baseline algorithms
    method_grid_=tuple(
        meth for meth in method_grid if meth in [
            'TPE_Tmin-PLT', 'ES', 'HB', 'TPE_Tmin', 'TPE_Tmin-Tmax', 'TPE_Tmin-ES'
        ]   
    )
    if method_grid_ and tune_criterion_grid_ and depth_trees_only_grid_: 
        configs += _get_run_experiment_configs(
            tune_criterion_grid=tune_criterion_grid_,
            depth_trees_only_grid=depth_trees_only_grid_,
            method_grid=method_grid_,
            scale_factor_grid=(scale_factor_grid[0],),
            delta_grid=(delta_grid[0],),
            n_trials_grid=(n_trials_grid[-1],),
            **kwargs,
        )

    # vary sf for ladder-based algorithms (plateau-like + Hyperband)
    method_grid_=tuple(meth for meth in method_grid if meth == 'PLATEAU')
    if method_grid_ and tune_criterion_grid_ and depth_trees_only_grid_ and len(scale_factor_grid) > 1:
        configs += _get_run_experiment_configs(
            tune_criterion_grid=tune_criterion_grid_,
            depth_trees_only_grid=depth_trees_only_grid_,
            method_grid=method_grid_,
            scale_factor_grid=scale_factor_grid[1:],
            delta_grid=(delta_grid[0],),
            n_trials_grid=(n_trials_grid[-1],),
            **kwargs,
        )

    # vary delta for plateau-like algorithms
    method_grid_=tuple(meth for meth in method_grid if meth == 'PLATEAU')
    if method_grid_ and tune_criterion_grid_ and depth_trees_only_grid_ and len(delta_grid) > 1:
        configs += _get_run_experiment_configs(
            tune_criterion_grid=tune_criterion_grid_,
            depth_trees_only_grid=depth_trees_only_grid_,
            method_grid=method_grid_,
            scale_factor_grid=(scale_factor_grid[0],),
            delta_grid=delta_grid[1:],
            n_trials_grid=(n_trials_grid[-1],),
            **kwargs,
        )

    return configs


# ---------- PARALLEL EXECUTION ----------

def process_dataset(
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    configs: Optional[List[Dict[str, Any]]] = None,
    dataset: str = "",
    random_state: Optional[int] = None,
    files_done: Optional[Sequence[Union[str, Path]]] = None,
    sort_experiments_by: Optional[Sequence[EXPERIMENT_SORT_RULE]] = None,
    # run_queue_pinned params
    use_smt: bool = True,
    # --- FileMover parameters
    move_dir: Optional[str] = None,
    move_min_age: int = 10,
    move_check_interval: int = 1,
    move_ignore_ext: str = '.tmp',
    # --- other control parameters
    progress_bar: bool = True,
    verbose: int = 0,
    **kwargs,
):
    """
    Build and run a queue of Random Forest HPO experiments for one dataset.

    This function is the dataset-level orchestration entry point. It either uses
    a user-provided list of experiment configurations (`configs`) or generates
    them from the grid parameters passed through `**kwargs`. The resulting
    configurations are split into shared (`common_params`) and per-run
    (`param_list`) arguments, optionally sorted, and then executed with
    `run_queue_pinned()`.

    The function can also start a background `FileMoverThread` to move completed
    log and dill files from the dataset directory to another location while the
    experiment queue is still running.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix passed to every experiment.
    y : np.ndarray
        Target vector passed to every experiment.
    problem : {'clf', 'reg'}
        Problem type: classification ('clf') or regression ('reg').
    score_func : callable
        OOB scoring function with signature ``score_func(y, y_pred_like) -> float``.
        It is forwarded to the selected tuning routine through `run_experiment()`.
    greater_is_better : bool
        Optimization direction for the scoring function. Use True for metrics to
        maximize and False for losses/errors to minimize.

    configs : list of dict or None, default=None
        Explicit per-experiment configuration dictionaries. If provided, these
        configurations are used directly and no configuration grid is generated.
        If None, configurations are generated by `_get_dataset_configs()` from
        the grid-related values passed through `**kwargs`.

    dataset : str, default=""
        Dataset name or dataset output directory. It is passed to generated
        configurations and is also used as the source directory for temporary
        files when `move_dir` is enabled.

    random_state : int or None, default=None
        Base random seed used when generating experiment configurations. If not
        None, generated runs typically receive consecutive seeds
        ``random_state + K`` across repeated experiments.

    files_done : sequence of str or Path or None, default=None
        Optional resume list of temporary ``.dill.tmp`` output paths to skip.
        Intended for restarting after timeout/crash without rerunning already
        completed experiments. The mechanism is filename-based and should be
        used with a fixed ``random_state``; with ``random_state=None``, generated
        timestamp-based seeds make output filenames non-reproducible across
        reruns.

    sort_experiments_by : sequence of {'random_state', 'plateau_first'} or None, default=None
        Optional ordering rules for the final experiment queue. The order of
        entries in this sequence defines sorting priority.

        Supported rules are:
        - ``"random_state"``: group/sort experiments by their `random_state`.
          This helps cover repeated random seeds more uniformly across methods.
        - ``"plateau_first"``: run plateau-like methods earlier. Currently this
          prioritizes ``"PLATEAU"`` and ``"TPE_Tmin-PLT"``.

        Examples:
        - ``("random_state", "plateau_first")`` keeps random states as the
          primary ordering and places plateau-like methods first inside each
          random-state block.
        - ``("plateau_first", "random_state")`` starts all plateau-like methods
          first and sorts them by random state internally.
        - None leaves the generated queue order unchanged.

    use_smt : bool, default=True
        Passed to `run_queue_pinned()`. If True, logical CPUs from SMT/hyperthreading
        may be used according to the CPU-pinning scheduler policy.

    move_dir : str or None, default=None
        Destination directory for completed output files. If not None and different
        from `dataset`, a background file mover is started. The source directory
        is cleaned from previous temporary files before the run starts.

    move_min_age : int, default=10
        Minimum file age in seconds before the background mover may move a file.

    move_check_interval : int, default=1
        Polling interval in seconds for the background file mover.

    move_ignore_ext : str, default='.tmp'
        File extension ignored by the background mover. Temporary files are renamed
        after completion, so this prevents moving incomplete outputs.

    progress_bar : bool, default=True
        Whether to show progress bars for the main experiment queue and, if enabled,
        the background file mover.

    verbose : int, default=0
        Verbosity level. The value passed to individual tuning routines is reduced
        via ``max(verbose - 3, 0)`` so that queue-level verbosity can be separated
        from per-trial logging.

    **kwargs
        Additional keyword arguments. Arguments accepted by `run_queue_pinned()`
        are forwarded to the scheduler. Important CPU-pinning controls include
        ``n_phys_cores_per_run`` (physical cores allocated to one experiment),
        ``socket_policy`` (``"prefer"``, ``"strict"``, or ``"none"``), and
        ``override_n_jobs``. By default, the scheduler injects ``n_jobs`` from the
        allocated CPU block; if a manually supplied ``n_jobs`` should be preserved,
        pass ``override_n_jobs=False``. The remaining arguments are used to build
        experiment configurations through `_get_dataset_configs()` when
        `configs is None`.

    Raises
    ------
    ValueError
        If `sort_experiments_by` contains unknown or duplicate sorting rules.
    ValueError
        If `move_dir` is a subdirectory of `dataset`, which would make the file
        mover recursively move files into its own source tree.

    Notes
    -----
    This function does not return experiment results directly. Each run writes a
    `.log` file and a `.dill` file via `run_experiment()`. The final analysis is
    expected to load those saved dill files.
    """
    cpu_pinning_args = _filter_kwargs(run_queue_pinned, kwargs)
    cpu_pinning_args['use_smt'] = use_smt
    cpu_pinning_args['progress_bar'] = progress_bar
    cpu_pinning_args['progress_bar_position'] = 0

    if configs is None:
        config_args = dict(kwargs)
        config_args['dataset'] = dataset
        config_args['random_state'] = random_state
        for k in list(cpu_pinning_args.keys()):
            config_args.pop(k, None)

        configs_ = _get_dataset_configs(**config_args)
    else:
        configs_ = configs

    files_done_set = None
    if files_done is not None:
        files_done_set = {Path(f) for f in files_done}

    common_params, param_list = split_common_params(configs_)
    common_params = {
        **common_params, 
        'X': X, 'y': y, 
        'problem': problem, 
        'score_func': score_func, 
        'greater_is_better': greater_is_better,
        'verbose': max(verbose - 3, 0),
        'files_done': files_done_set,
    }

    # Sort param_list according to sort_experiments_by
    if sort_experiments_by is not None:
        plateau_methods = {"PLATEAU", "TPE_Tmin-PLT"}
        allowed = set(get_args(EXPERIMENT_SORT_RULE))

        unknown = set(sort_experiments_by) - allowed
        if unknown:
            raise ValueError(
                f"Unknown sort_experiments_by rules: {unknown}. "
                f"Allowed rules are: {sorted(allowed)}."
            )

        if len(set(sort_experiments_by)) != len(tuple(sort_experiments_by)):
            raise ValueError(
                f"Duplicate sort_experiments_by rules are not allowed: {sort_experiments_by}."
            )

        sort_key = {
            "random_state": lambda p: (
                p.get("random_state", common_params.get("random_state"))
                if p.get("random_state", common_params.get("random_state")) is not None
                else float("-inf")
            ),
            "plateau_first": lambda p: (
                0 if p.get("method", common_params.get("method")) in plateau_methods else 1
            ),
        }

        param_list = [
            p for _, p in sorted(
                enumerate(param_list),
                key=lambda item: (
                    tuple(sort_key[rule](item[1]) for rule in sort_experiments_by)
                    + (item[0],)
                ),
            )
        ]

    # Start background file mover if requested
    if move_dir is not None and dataset != move_dir:
        # Prevent moving into a subdirectory of the source (would cause recursion)
        if Path(dataset).resolve() in Path(move_dir).resolve().parents:
            raise ValueError(f"{move_dir} can not be a subdirectory of {dataset}!")

        # Clean source directory before starting (all previous temporary files are removed)
        clean_source_dir(dataset, confirm=False, verbose=verbose)

        # Count total files that will be generated:
        # - Each method name component (separated by '-') corresponds to one log file.
        #   e.g., "TPE" → 1 log, "TPE_Tmin-PLT" → 2 logs.
        # - Each run produces exactly one dill file.
        n_log_files = 0
        for p in param_list:
            method_name = p.get("method") or common_params["method"]
            n_log_files += len(method_name.split('-'))
        n_dill_files = len(param_list) 

        mover = FileMoverThread(
            source_dir=dataset,
            dest_dir=move_dir,
            min_age=move_min_age,
            check_interval=move_check_interval, 
            ignore_ext=move_ignore_ext, 
            total_files=n_log_files + n_dill_files,
            progress_bar=progress_bar,
            progress_bar_position=1,
            autostart=True,
            verbose = verbose,
        )
    else:
        mover = None

    # run main queue
    cpu_pinning_args['verbose'] = verbose
    run_queue_pinned(
        run_experiment,
        param_list,
        static_params=common_params,
        return_outputs=False, 
        **cpu_pinning_args,
    )

    if mover is not None:
        # In resume mode, some experiments are skipped via files_done, so the
        # precomputed total_files may include outputs that will never be created.
        # Do not wait indefinitely for skipped runs.
        resume_mode = bool(files_done)
        mover.wait_for_completion(timeout=move_min_age + 2 * move_check_interval if resume_mode else None)
        mover.stop()


