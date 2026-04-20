"""
run_experiments.py
------------------

This module provides functions for running Random Forest tuning experiments using
three different search methods implemented in `rf_plateau_hpo.core`:

- `tune_rf_oob`          – classic Optuna HPO (baseline).
- `tune_rf_oob_bohb`     – BOHB‑like multi‑fidelity baseline (TPE + Hyperband).
- `tune_rf_oob_plateau`  – the proposed plateau‑search algorithm (new method).

All experiments are logged and their results (study summaries, best configurations,
metadata) are saved as `.dill` files for later analysis. The module also includes
helpers to generate parameter grids and to orchestrate large‑scale parallel runs
using `cpu_pinning.run_queue_pinned` for CPU‑affinity‑controlled parallel execution.

Key Functions
-------------
- `run_experiment()`          – Run a single tuning experiment with a given method.
- `process_dataset()`         – Generate a full factorial set of experiment configurations
                                and execute them in parallel with CPU pinning.
- `_get_run_experiment_configs()` – Build a list of configuration dicts for all combinations
                                    of hyperparameters (method, delta, n_trials, etc.).
- `_get_dataset_configs()`    – Higher‑level grid builder that creates configurations
                                for baseline comparisons and sensitivity analyses.

Requirements
------------
- The main package `rf_plateau_hpo` (with its dependencies: numpy, pandas, scikit‑learn,
  optuna, PyYAML) must be installed.
- **Additional required package**: `dill` (≥0.3.8) for serialisation of experiment results.
- The CPU‑pinning scheduler (`cpu_pinning`) relies on Linux-specific system calls
  (`os.sched_setaffinity`); it will work only on Linux (on other OSes it falls back
  to `spawn` start method without affinity control).

All functions share consistent logging and parameter handling. For analysing the
saved results, see the companion module `analyze_experiments.py`.

Copyright (c) 2025 Andrey Lange
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


def _parse_study(study: optuna.Study) -> Dict[str, Any]:
    """
    Summarize an Optuna Study primarily using user_attrs produced by our objectives.
    Additionally, include BEST_* fields (trial number, score, params) from `study.best_trial`.
    """
    trials = list(study.trials)

    out = Counter()
    out["n_trials_total"] = len(trials)

    n_trials_with_trees_built = 0
    for t in trials:
        ua = t.user_attrs

        if "trees_built" in ua:
            out["n_trees_built"] += int(ua["trees_built"])
            n_trials_with_trees_built += 1

        if "pruned" in ua:
            out["n_trials_pruned_"+str(ua["pruned"])] += 1
            out["n_trials_pruned"] += 1

        if "shift" in ua:
            sh = int(ua["shift"])
            if sh == -1:
                out["n_trials_shift_left"] += 1
            if sh == 0:
                out["n_trials_stay"] += 1
            if sh == 1:
                out["n_trials_shift_right"] += 1
                if "pruned" in ua:
                    out["n_trials_shift_right_pruned_"+str(ua["pruned"])] += 1

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
def _parse_log_tail(log_file: Union[str, Path]) -> Dict[str, Any]:
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


def _sumup_common(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
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
    note: str = ""
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

    # if dill_file in DOROTHEA_FILES:
    #     return

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
        return {**_parse_log_tail(log_file), **_parse_study(study), 'study': study, 'log_file': log_file}

    study0, study = None, None
    log_file0 = None
    if "TPE_Tmin" in method:
        # Special baselines when n_estimators and other hyperparameters are tuned separetely
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
                    f"In '{method}' it is asumed that n_estimators_ladder[0]=Tmin."
                    f"(n_estimators_ladder[0]={n_estimators_ladder[0]}, Tmin={t_min})",
                    RuntimeWarning,
                )

            if hyperband_reduction_factor >= 2:
                warnings.warn(
                    f"In '{method}' it is asumed that hyperband_reduction_factor < 2 (no pruning)."
                    f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                    RuntimeWarning,
                )

            _, _, study, _ = tune_rf_oob_bohb(**{**bohb_params, **fixed_params})

        elif method == "TPE_Tmin-PLT":
            # TPE Tmin-PLT: classic HPO search with fixed T=t_min + plateau search 
            if n_estimators_start != t_min:
                warnings.warn(
                    f"In '{method}' it is asumed that n_estimators_start=Tmin."
                    f"(n_estimators_start={n_estimators_start}, Tmin={t_min})",
                    RuntimeWarning,
                )

            _, _, study, _ = tune_rf_oob_plateau(**{**plateau_params, **fixed_params})

    elif method == "TPE":
        # TPE: classic search over all hyperparameters
        _, study = tune_rf_oob(**classic_params)
    elif method == "ES":
        # ES: early-stoping with delta tolerance
        if hyperband_reduction_factor >= 2:
            warnings.warn(
                f"In '{method}' it is asumed that hyperband_reduction_factor < 2 (no pruning)."
                f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                RuntimeWarning,
            )

        _, _, study, _ = tune_rf_oob_bohb(**bohb_params)
    elif method == "HB":
        # HB: Hyperband-like with n_estimators budgets 
        if hyperband_reduction_factor < 2:
            warnings.warn(
                f"In '{method}' (Hyperband) it is asumed that hyperband_reduction_factor >= 2 (HyperbandPruner() is used)."
                f"(hyperband_reduction_factor={hyperband_reduction_factor})",
                RuntimeWarning,
            )
        
        if delta >= 0.0:
            warnings.warn(
                f"In '{method}' (Hyperband) it is asumed that delta < 0 (no early stopping)."
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

    output = _sumup_common(out0, out)
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
    def _build_ladder(scale_factor: float, n_estimators_start: int, t_max: int):
        ladder = [n_estimators_start]
        ladder += [int(round(n_estimators_start * scale_factor))]
        
        while ladder[-1] < t_max:
            if ladder[-2] >= ladder[-1]:
                raise ValueError("n_estimators_ladder must be strictly increasing.")

            ladder.append(int(round(ladder[-1] * scale_factor)))
        return tuple(ladder)

    configs = []
    for tune_criterion in tune_criterion_grid:
        for depth_trees_only in depth_trees_only_grid:
            for method in method_grid:
                for scale_factor in scale_factor_grid:

                    ladder = _build_ladder(scale_factor, n_estimators_start, max_trees if method in ["TPE_Tmin-ES", "ES"] else t_max)
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
    method_grid_ = tuple(meth for meth in ['TPE', 'PLATEAU'] if meth in method_grid)
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
        meth for meth in [
            'TPE_Tmin', 'TPE_Tmin-Tmax', 'TPE_Tmin-ES', 'TPE_Tmin-PLT', 'ES', 'HB'
        ] if meth in method_grid 
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
    method_grid_=tuple(
        meth for meth in [
            'PLATEAU'
        ] if meth in method_grid 
    )
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
    method_grid_=tuple(
        meth for meth in [
            'PLATEAU'
        ] if meth in method_grid 
    )
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
    sort_params_by_random_state = False,
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

    common_params, param_list = split_common_params(configs_)

    common_params = {
        **common_params, 
        'X': X, 'y': y, 
        'problem': problem, 
        'score_func': score_func, 
        'greater_is_better': greater_is_better,
        'verbose': max(verbose - 3, 0),
    }

    if sort_params_by_random_state and random_state is not None:
        param_list.sort(key=lambda p: p.get('random_state', float('-inf')))

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
        mover.wait_for_completion()
        mover.stop()


# with open('dorothea_files.txt', 'r', encoding='utf-8') as f:
#     DOROTHEA_FILES = [line.strip() for line in f if line.strip()]
# DOROTHEA_FILES = [Path(f.replace('/home/a.lange/RF_plateau_HPO/rf_plateau_hpo/notebooks', '/tmp/a.lange') + '.tmp') for f in DOROTHEA_FILES]