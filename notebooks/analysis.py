"""
analysis.py
-----------
This module contains helper functions for running and analyzing experiments related to
the `rf_plateau_hpo` library, specifically for tuning Random Forest models using Out-Of-Bag (OOB)
and Plateau-based tuning methods (i.e., `tune_rf_oob()` and `tune_rf_oob_plateau()`).

Functions in this module allow the execution of experiments with customizable hyperparameters,
logging of experiment details, and the analysis of results.

Key Functions:
--------------
1. `run_experiment()`: Runs an experiment with a given Random Forest tuning method (`tune_rf_oob` or `tune_rf_oob_plateau`).
   Logs the results, parses the best-performing configurations, and stores experiment details in a `dill` file.
   
2. `process_dataset()`: Automates the execution of experiments for a given dataset, iterating over various hyperparameter configurations.
   
3. `analyze_experiment_results()`: Analyzes the results of experiments stored in `.dill` files. Computes statistics
   (mean, std) and generates visualizations (boxplots) for key metrics like BEST scores, total time, and n_estimators.

4. `compare_experiment_groups()`: Compares results between different experiment groups using statistical tests (Mann-Whitney U).
   
5. `plot_dataset_comparisons()`: Plots comparisons of key metrics (e.g., total time, BEST scores, n_estimators) between experiment groups.

Requires:
---------
- "dill>=0.3.8"
- "typing_extensions>=3.10"
- "matplotlib>=3.5"
- "seaborn>=0.11.2"
- Dependencies from `rf_plateau_hpo` library (specified in pyproject.toml)
"""
import ast
from datetime import datetime, timezone
from inspect import Parameter, signature
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    # Python 3.8+ 
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal

import os
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

sns.set(style="whitegrid")

# --- Local / application ---
from rf_plateau_hpo.core import tune_rf_oob, tune_rf_oob_plateau

# ---------- EXPERIMENT RUNNING ----------

# Main experiment-runner function 
def run_experiment(
    method: Callable,
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    # --- Output directory ---
    outdir: Union[str, Path] = "",

    # --- Search space ---
    max_features_grid: Sequence[object] = ("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0),
    max_depth_range: Tuple[int, int] = (4, 40),
    min_samples_leaf_range: Tuple[int, int] = (1, 20),
    min_samples_split_range: Tuple[int, int] = (2, 40),
    tune_criterion: bool = True,

    # --- Pass to Random Forest class ---
    criterion: Optional[str] = None,
    class_weight: Optional[Union[str, Dict[str, float], List[Dict[str, float]]]] = None,

    # --- Optuna / runtime ---
    n_trials: int = 40,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: int = 0,

    # --- tune_rf_oob ---
    # ignored if method == tune_rf_oob_plateau
    n_estimators_range: Optional[Tuple[int, int]] = (50, 2000),

    # --- tune_rf_oob_plateau ---
    # ignored if method == tune_rf_oob
    n_estimators_start: Optional[int] = 100,
    scale_factor: Optional[float] = 1.5,
    delta: Optional[float] = 2e-3,
    max_trees: Optional[int] = 5000,
    
    # --- info ---
    dataset: str = "",
    note: str = ""
) -> Dict[str, Any]:
    """
    Runs an experiment, logs BEST-lines to a file, parses the contiguous BEST tail,
    and stores a dill artifact with inputs/outputs/parsed params.
    """    
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)    
 
    ts = datetime.now(timezone.utc)
    random_state_ = int(ts.timestamp()) if random_state is None else random_state

    run_id = '-'.join([method.__name__, dataset, str(n_trials), str(random_state_)])
    log_file = outdir / f"{run_id}.log"
    dill_file = outdir / f"{run_id}.dill"
    
    params = {
        'X': X,
        'y': y,
        'problem': problem,
        'score_func': score_func,
        'greater_is_better': greater_is_better,
        'max_features_grid': max_features_grid,
        'max_depth_range': max_depth_range,
        'n_estimators_range': n_estimators_range,
        'min_samples_leaf_range': min_samples_leaf_range,
        'min_samples_split_range': min_samples_split_range,
        'tune_criterion': tune_criterion,
        'criterion': criterion,
        'class_weight': class_weight,
        'n_estimators_start': n_estimators_start,
        'scale_factor': scale_factor,
        'delta': delta,
        'max_trees': max_trees,
        'n_trials': n_trials,
        'n_jobs': n_jobs,
        'random_state': random_state_,  # note: key 'random_state' maps to variable random_state_
        'verbose': verbose,
        'log_file': log_file,
        'refit': False,
    }
    
    if method == tune_rf_oob:
        params_oob_plateau_only = TUNE_RF_OOB_PLATEAU_PARAMS_DFLT.keys() - TUNE_RF_OOB_PARAMS_DFLT.keys()
        params = {k: v for k, v in params.items() if k not in params_oob_plateau_only}
        _, study = tune_rf_oob(**params)
        out = {'study': study}
        log_parsed = parse_log_tail(log_file)
    else:
        params_oob_only = TUNE_RF_OOB_PARAMS_DFLT.keys() - TUNE_RF_OOB_PLATEAU_PARAMS_DFLT.keys()
        params = {k: v for k, v in params.items() if k not in params_oob_only}
        _, best_n_estimators, study, plateau_found = tune_rf_oob_plateau(**params)
        out = {
            'best_n_estimators': best_n_estimators, 
            'study': study,
            'plateau_found': plateau_found
        }
        log_parsed = parse_log_tail(log_file)
    
    del params['X'], params['y']
    
    data_params = {'dataset': dataset}
    if hasattr(X, 'shape'):
        data_params['X.shape'] = X.shape
    if hasattr(X, 'columns'):
        data_params['X.columns'] = X.columns
    if hasattr(y, 'name'):
        data_params['y.name'] = y.name
    
    run_params = {
        'note': note,
        'method': method.__name__,
        'experiment_started': ts.strftime("%Y-%m-%d %H:%M:%S"),
        'params_data': data_params,
        'params_in': params,
        'params_out': out,
        'params_parsed': log_parsed,
    }
    
    with open(dill_file, "wb") as f:
        dill.dump(run_params, f, protocol=dill.HIGHEST_PROTOCOL)

    if verbose > 0:
        print("Log file:", log_file)
        print("Pickle file:", dill_file)
    
    return run_params, log_file, dill_file


def process_dataset(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    tune_criterion_range: List[bool] = [True, False],
    depth_trees_only_range: List[bool] = [True, False],
    method_range: List[Callable] = [tune_rf_oob, tune_rf_oob_plateau],
    delta_range: List[float] = [2e-3],
    **run_experiment_args
):
    for tune_criterion in tune_criterion_range:
        for depth_trees_only in depth_trees_only_range:
            for method in method_range:
                for delta in delta_range if method == tune_rf_oob_plateau else [None]:
                    kwargs = {k:v for k, v in run_experiment_args.items()}
                    kwargs['tune_criterion'] = tune_criterion
                    if depth_trees_only: # "only depth and trees": tune only max_depth and n_estimators
                        kwargs.update({
                            'min_samples_leaf_range': (1, 1),
                            'min_samples_split_range': (2, 2),
                            'max_features_grid': ('sqrt',), 
                        })
                    if delta is not None:
                        kwargs['delta'] = delta
                        
                    for n_trials in [40, 120]:
                        outdir = Path(dataset)
                        outdir = (outdir / f"tune_criterion={tune_criterion}")
                        outdir = (outdir / f"depth_trees_only={depth_trees_only}")
                        outdir = (outdir / method.__name__)
                        if 'delta' in kwargs:
                            outdir = (outdir / f"delta={kwargs['delta']:.0e}".replace('e-0', 'e-'))
                        outdir = (outdir / f"n_trials={n_trials}")

                        for K in range(20):
                            run_params, log_file, dill_file = run_experiment(
                                method=method,
                                X=X, y=y, 
                                problem=problem,
                                score_func = score_func,
                                greater_is_better=greater_is_better,
                                n_trials=n_trials,
                                dataset=dataset,
                                verbose=0,
                                outdir=outdir,
                                note=str(K),
                                **kwargs                    
                            )

# ---------- ANALYZE & VISUALIZE EXPERIMENTS ----------

# get function signature
def declared_params(func):
    sig = signature(func)  # no execution of the function body
    rows = []
    for name, p in sig.parameters.items():
        rows.append({
            "name": name,
            "kind": p.kind.name,  # POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD
            "default": None if p.default is Parameter.empty else p.default,
            "annotation": None if p.annotation is Parameter.empty else p.annotation,
        })
    ret_ann = None if sig.return_annotation is Parameter.empty else sig.return_annotation
    return rows, ret_ann

# def f(a, /, b: int, *args, c=3, d: float = 1.5, **kw) -> str:
#     return "ok"
# print(declared_params(f)[0])


TUNE_RF_OOB_PARAMS_DFLT = {p['name']: p['default'] for p in declared_params(tune_rf_oob)[0]}
TUNE_RF_OOB_PLATEAU_PARAMS_DFLT = {p['name']: p['default'] for p in declared_params(tune_rf_oob_plateau)[0]}

# parse the end-block of the log-file and extract `BEST'-parameters
def parse_log_tail(log: Union[str, Path, Sequence[str]]) -> Dict[str, Any]:
    if isinstance(log, Path):
        lines = log.read_text(encoding="utf-8").splitlines()
    elif isinstance(log, str):
        p = Path(log)
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() and ("\n" not in log and "\r" not in log) else log.splitlines()
    else:
        lines = [str(x) for x in log]
    if not lines:
        return {}

    def is_best(s: str) -> bool:
        parts = [c.strip() for c in s.split("|")]
        return len(parts) >= 3 and parts[2].startswith("BEST")

    block = []
    for ln in reversed(lines):
        if is_best(ln):
            block.append(ln)
        elif block:
            break
    block.reverse()
    if not block:
        return {}

    out: Dict[str, Any] = {}
    time_re = re.compile(r"\[t\+(\d+(?:\.\d+)?)s\]\s*$")

    for line in block:
        line = time_re.sub("", line)
        cols = [c.strip() for c in line.split("|")]
        for seg in (c.strip() for c in cols[2:] if len(cols) >= 3):
            if not seg:
                continue
            body = seg[5:].strip() if seg.startswith("BEST ") else (seg[4:].strip() if seg.startswith("BEST") else seg)
            if "=" not in body:
                continue
            k, v = body.split("=", 1)
            try:
                val: Any = ast.literal_eval(v.strip())
            except Exception:
                val = v.strip()
            out[f"BEST {k.strip()}"] = val

    m = time_re.search(block[-1])
    if m:
        out["Total time"] = float(m.group(1))
    return out


def analyze_experiment_results(
    folder_path: str, 
    save_plots: bool = True, 
    show_plots: bool = False
) -> Dict[str, float]:
    # Dictionary for storing results
    results = {'BEST scores': [], 'Total time': [], 'n_estimators': [], 'max_depth': []}
    results.update({
        'problem': None,
        'greater_is_better': None,
        'BEST scores mean': None,
        'BEST scores std': None,
        'Total time mean': None,
        'Total time std': None,
        'n_estimators mean': None,
        'n_estimators std': None,
    })

    # Reading all .dill files in the folder
    dill_files = list(Path(folder_path).glob("*.dill"))
    if len(dill_files) == 0:
        return results
    
    for dill_file in dill_files:
        with open(dill_file, 'rb') as f:
            experiment = dill.load(f)

        try:
            # Extracting the necessary data from the dictionary
            problem = experiment['params_in']['problem']
            greater_is_better = experiment['params_in']['greater_is_better']
            best_score = experiment['params_parsed']['BEST score']
            total_time = experiment['params_parsed']['Total time']
            max_depth = experiment['params_parsed']['BEST params']['max_depth']

            # Checking where to get the number of trees (n_estimators)
            if 'best_n_estimators' in experiment.get('params_out', {}):
                n_estimators = experiment['params_out']['best_n_estimators']
            else:
                n_estimators = experiment['params_parsed']['BEST params']['n_estimators']

            # Adding the data to the results
            results['problem'] = problem
            results['greater_is_better'] = greater_is_better
            results['BEST scores'].append(best_score)
            results['Total time'].append(total_time)
            results['n_estimators'].append(n_estimators)
            results['max_depth'].append(max_depth)
        except KeyError as e:
            print(f"Key error in file {dill_file}: {e}")

    # Converting data to numpy arrays for statistics
    best_scores = np.array(results['BEST scores'])
    total_times = np.array(results['Total time'])
    n_estimators = np.array(results['n_estimators'])
    max_depth = np.array(results['max_depth'])

    # Adding statistics to results
    results.update({
        'BEST scores mean': np.mean(best_scores),
        'BEST scores std': np.std(best_scores),
        'Total time mean': np.mean(total_times),
        'Total time std': np.std(total_times),
        'n_estimators mean': np.mean(n_estimators),
        'n_estimators std': np.std(n_estimators),
    })

    # Function for saving and displaying boxplots with individual points
    def save_boxplot(data, title, filename):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        plt.scatter(np.ones(len(data)), data, color='red', alpha=0.5)  # Mark individual points with red dots
        plt.title(title)
        if save_plots:
            plt.savefig(filename, bbox_inches="tight")
        if show_plots:
            plt.show()  # Display the plot in the notebook
        plt.close()

    # Saving and displaying the boxplots
    save_boxplot(best_scores, "BEST Scores", os.path.join(folder_path, "best_scores.jpg"))
    save_boxplot(total_times, "Total Time", os.path.join(folder_path, "total_time.jpg"))
    save_boxplot(n_estimators, "BEST n_estimators", os.path.join(folder_path, "n_estimators.jpg"))

    # Scatter plot for n_estimators vs max_depth
    plt.figure(figsize=(10, 6))
    plt.scatter(n_estimators, max_depth, c='blue', alpha=0.5)
    plt.title("n_estimators vs max_depth")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")

    # Ensure that n_estimators and max_depth axes have integer ticks
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if save_plots:
        plt.savefig(os.path.join(folder_path, "n_estimators_vs_max_depth.jpg"), bbox_inches="tight")
    if show_plots:
        plt.show()  # Display the plot in the notebook
    plt.close()

    return results


def compare_experiment_groups(dataset_folder: str, delta_dir: str="delta=2e-3", save_plots: bool=True):
 
    def check(path1: str, path2: str, param: str, alternative: str, save_plots: bool=True) -> float:
        exp_res1 = analyze_experiment_results(path1, save_plots=save_plots)
        exp_res2 = analyze_experiment_results(path2, save_plots=save_plots)

        if exp_res1[param] is None or exp_res2[param] is None:
            return ""

        alt = alternative
        if param == "BEST scores":
            gib1 = exp_res1['greater_is_better']
            gib2 = exp_res2['greater_is_better']
            if gib1 is None or gib2 is None:
                return ""

            assert(gib1==gib2)
            if not gib1:
                if alt=='greater':
                    alt='less'
                elif alt=='less':
                    alt='greater'

        return stats.mannwhitneyu(exp_res1[param], exp_res2[param], alternative=alt)[1]

    results = []  # Store comparison results

    for tune_criterion in [False, True]:
        for depth_trees_only in [True, False]:
            for group_1, group_2 in [(120, 40)]:
                path1 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob' / f'n_trials={group_1}'
                path2 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob' / f'n_trials={group_2}'
                p_value = check(path1, path2, 'BEST scores', 'greater', save_plots=save_plots)
                results.append({'tune_criterion': tune_criterion, 'depth_trees_only': depth_trees_only, 'function': 'tune_rf_oob', 
                    'n_trials': f"Score({group_1}) (better ?)\nScore({group_2})", 'p_value': p_value})

    for tune_criterion in [False, True]:
        for depth_trees_only in [True, False]:
            for group_1, group_2 in [(120, 40)]:
                path1 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob_plateau' / f'{delta_dir}' / f'n_trials={group_1}'
                path2 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob_plateau' / f'{delta_dir}' / f'n_trials={group_2}'
                p_value = check(path1, path2, 'BEST scores', 'greater', save_plots=save_plots)
                results.append({'tune_criterion': tune_criterion, 'depth_trees_only': depth_trees_only, 'function': 'tune_rf_oob_plateau', 
                    'n_trials': f"Score({group_1}) (better ?)\nScore({group_2})", 'p_value': p_value})

    for tune_criterion in [False, True]:
        for depth_trees_only in [True, False]:
            path1 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob' / 'n_trials=120'
            path2 = Path(dataset_folder) / f'tune_criterion={tune_criterion}' / f'depth_trees_only={depth_trees_only}' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
            p_value = check(path1, path2, 'BEST scores', 'greater', save_plots=save_plots)
            results.append({'tune_criterion': tune_criterion, 'depth_trees_only':depth_trees_only, 'n_trials': 120, 
                'function': 'Score(tune_rf_oob) (better ?)\nScore(tune_rf_oob_plateau)', 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=True'  / 'tune_rf_oob' / 'n_trials=120'
    p_value = check(path1, path2, 'BEST scores', 'greater', save_plots=save_plots)
    results.append({'tune_criterion': False, 'function': 'tune_rf_oob', 'n_trials': 120, 
        'depth_trees_only': 'Score(False) (better ?)\nScore(True)', 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=True'  / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    p_value = check(path1, path2, 'BEST scores', 'greater', save_plots=save_plots)
    results.append({'tune_criterion': False, 'function': 'tune_rf_oob_plateau', 'n_trials': 120, 
        'depth_trees_only': 'Score(False) (better ?)\nScore(True)', 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=True' / 'tune_rf_oob' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=True'  / 'depth_trees_only=True' / 'tune_rf_oob' / 'n_trials=120'
    p_value = check(path1, path2, 'BEST scores', 'less', save_plots=save_plots)
    results.append({'depth_trees_only': True, 'function': 'tune_rf_oob', 'n_trials': 120, 
        'tune_criterion': 'Score(False) (better ?)\nScore(True)', 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=True' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=True' / 'depth_trees_only=True' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    p_value = check(path1, path2, 'BEST scores', 'less', save_plots=save_plots)
    results.append({'depth_trees_only': True, 'function': 'tune_rf_oob_plateau', 'n_trials': 120, 
        'tune_criterion': f"Score(False) (better ?)\nScore(True)", 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    p_value = check(path1, path2, 'Total time', 'greater', save_plots=save_plots)
    results.append({'tune_criterion': False, 'depth_trees_only': False, 'n_trials': 120, 
        'function': f"Time(tune_rf_oob) (better ?)\nTime(tune_rf_oob_plateau)", 'p_value': p_value})

    path1 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob' / 'n_trials=120'
    path2 = Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob_plateau' / f'{delta_dir}' / 'n_trials=120'
    p_value = check(path1, path2, 'n_estimators', 'greater', save_plots=save_plots)
    results.append({'tune_criterion': False, 'depth_trees_only': False, 'n_trials': 120, 
        'function': f"n_estimators(tune_rf_oob) (better ?)\nn_estimators(tune_rf_oob_plateau)", 'p_value': p_value})

    # Convert the results to a DataFrame for easier viewing
    res = []
    for d in results:
        res.append(
            {('p_value', os.path.basename(Path(dataset_folder))) if k == 'p_value' else ('', k): v for k, v in d.items()}
        )
    columns = pd.MultiIndex.from_tuples(res[0].keys(), names=['', ''])
    return pd.DataFrame(res, columns=columns)


def plot_dataset_comparisons(
    dataset_folders: List[str],
    tune_criterion: bool = True,
    depth_trees_only: bool = False,
    delta_dir: str = "delta=2e-3",
    n_trials: int = 120,
    ncols: Optional[int] = None,
    show_scores: bool = False, # new parameter to control displaying BEST scores
    bw: float = 0.9,           # bar width (further reduced for closer spacing between bars of the same group)
                               # smaller width for tighter spacing between bars in each group
    min_w = 3.0,  # inches per subplot column
    min_h = 2.5,  # inches per subplot row
    tight_layout_top=0.9,
    save_plots=True
):
    """
    For each dataset folder, plot paired bar groups:
    - Total time (left y-axis, seconds)
    - n_estimators (right y-axis, integer ticks)

    For each metric we show 2 bars:
    - tune_rf_oob()          (solid fill)
    - tune_rf_oob_plateau()  (same color with hatch)

    Bars per subplot are arranged in three groups on the x-axis:
        [time_oob, time_plateau, best_oob, best_plateau, n_estim_oob, n_estim_plateau]

    The x-axis has no tick labels. Grid is removed.
    A shared legend is shown above all subplots.
    """

    n_dsets = len(dataset_folders)

    # ----- subplot grid shape -------------------------------------------------
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(n_dsets)))
    nrows = int(math.ceil(n_dsets / ncols))

    # Adjust the figure size based on the grid shape
    fig_w = ncols * min_w
    fig_h = nrows * min_h + 1.0  # +1.0in headroom for legend/title (slightly smaller)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))

    # normalize axes to 1D list
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel()
    else:
        axes_list = [axes]

    legend_handles = []
    legend_labels = []

    # ----- visual style config (consistent fonts etc.) ------------------------
    title_fs = 12          # dataset title font size
    axis_label_fs = 12     # y-axis label font size
    tick_fs = 11           # tick label font size
    legend_fs = 12         # legend font size

    # colors: one per metric
    color_time = "#4C72B0"   # Total time bars
    color_nest = "#C44E52"   # n_estimators bars
    color_scores = "#55A868"   # BEST scores bars

    hatch_plateau = "//"     # hatch style for plateau bars

    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[i]

        # get metrics for tune_rf_oob()
        res_oob = analyze_experiment_results(
            Path(dataset_folder)
            / f"tune_criterion={tune_criterion}"
            / f"depth_trees_only={depth_trees_only}"
            / "tune_rf_oob"
            / f"n_trials={n_trials}", save_plots=save_plots
        )

        # get metrics for tune_rf_oob_plateau()
        res_plateau = analyze_experiment_results(
            Path(dataset_folder)
            / f"tune_criterion={tune_criterion}"
            / f"depth_trees_only={depth_trees_only}"
            / "tune_rf_oob_plateau"
            / f"{delta_dir}"
            / f"n_trials={n_trials}", save_plots=save_plots
        )

        # extract stats
        # Total time
        time_mean_oob = res_oob["Total time mean"]
        time_std_oob  = res_oob["Total time std"]
        time_mean_pl  = res_plateau["Total time mean"]
        time_std_pl   = res_plateau["Total time std"]

        # n_estimators
        n_mean_oob = np.mean(res_oob["n_estimators"])
        n_std_oob  = np.std(res_oob["n_estimators"])
        n_mean_pl  = np.mean(res_plateau["n_estimators"])
        n_std_pl   = np.std(res_plateau["n_estimators"])

        # BEST scores
        best_scores_mean_oob = res_oob["BEST scores mean"]
        best_scores_std_oob  = res_oob["BEST scores std"]
        best_scores_mean_pl  = res_plateau["BEST scores mean"]
        best_scores_std_pl   = res_plateau["BEST scores std"]

        # x positions: group 1 = time, group 2 = best scores, group 3 = n_estimators
        # Leave a gap between groups
        if show_scores:
            x_time = np.array([0.0, 1.0])  # close to each other
            x_scores = np.array([2.5, 3.5])  # moved to the second position
            x_nest = np.array([5.0, 6.0])  # even more spaced out from best scores
        else:
            x_time = np.array([0.0, 1.0]) 
            x_nest = np.array([3.0, 4.0])

        # left axis: Total time
        bars_time = ax.bar(
            x_time,
            [time_mean_oob, time_mean_pl],
            yerr=[time_std_oob, time_std_pl],
            width=bw,
            color=[color_time, color_time],
            hatch=["", hatch_plateau],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_ylabel("Total time (s)", fontsize=axis_label_fs)
        ax.set_title(Path(dataset_folder).name, fontsize=title_fs)

        # second axis: n_estimators
        ax2 = ax.twinx()
        bars_nest = ax2.bar(
            x_nest,
            [n_mean_oob, n_mean_pl],
            yerr=[n_std_oob, n_std_pl],
            width=bw,
            color=[color_nest, color_nest],
            hatch=["", hatch_plateau],
            edgecolor="black",
            linewidth=0.8,
        )
        ax2.set_ylabel("n_estimators", fontsize=axis_label_fs)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # third axis: BEST scores (only if show_scores=True)
        if show_scores:
            ax3 = ax.twinx()  # Create a third axis for the third group
            # ax3.spines['right'].set_position(('outward', 60))  # Move the third axis outward
            bars_scores = ax3.bar(
                x_scores,
                [best_scores_mean_oob, best_scores_mean_pl],
                yerr=[best_scores_std_oob, best_scores_std_pl],
                width=bw,
                color=[color_scores, color_scores],
                hatch=["", hatch_plateau],
                edgecolor="black",
                linewidth=0.8,
            )
            # ax3.set_ylabel("BEST Scores", fontsize=axis_label_fs)
            ax3.yaxis.set_major_locator(plt.NullLocator())  # Remove labels from third axis

        # remove grid
        ax.grid(False)
        ax2.grid(False)
        if show_scores:
            ax3.grid(False)

        # hide all x tick labels
        if show_scores:
            x_all = np.concatenate([x_time, x_scores, x_nest])
        else:
            x_all = np.concatenate([x_time, x_nest])
        ax.set_xticks(x_all)
        ax.set_xticklabels([])
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax2.tick_params(axis="both", which="major", labelsize=tick_fs)

        # collect legend handles once, from first subplot
        if i == 0:
            # We want:
            #  time_oob  (color_time, no hatch)
            #  time_plateau (color_time, hatch)
            #  n_estim_oob  (color_nest, no hatch)
            #  n_estim_plateau (color_nest, hatch)
            #  best_scores_oob  (color_scores, no hatch)
            #  best_scores_plateau (color_scores, hatch)
            legend_handles.extend([
                bars_time[0],
                bars_time[1],
                bars_nest[0],
                bars_nest[1],
            ])
            if show_scores:
                legend_handles.extend([
                    bars_scores[0],
                    bars_scores[1],
                ])
            legend_labels.extend([
                "time tune_rf_oob()",
                "time tune_rf_oob_plateau()",
                "n_estimators tune_rf_oob()",
                "n_estimators tune_rf_oob_plateau()",
            ])
            if show_scores:
                legend_labels.extend([
                    "BEST scores tune_rf_oob()",
                    "BEST scores tune_rf_oob_plateau()",
                ])

    # hide any unused axes if grid > number of datasets
    for j in range(n_dsets, len(axes_list)):
        fig.delaxes(axes_list[j])

    # ----- global legend placement -------------------------------------------
    # Move the legend closer to the plots.
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),  # now closer to subplots
    )

    # Make layout tight but leave minimal margin for legend
    fig.tight_layout(rect=[0, 0, 1, tight_layout_top])  # smaller space for legend
    plt.show()

