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
   Runs `run_experiment()` with different control parameters for specified dataset.
   
3. `read_experiment_results()`: Analyzes the results of experiments stored in `.dill` files. 
   Computes statistics (mean, std) and generates visualizations (boxplots) for key metrics like BEST scores, total time, and n_estimators.

4. `compare_experiment_groups()`: Compares results between different experiment groups using statistical tests (t-test, Mann-Whitney U).

5. `process_html_table()`: Creates ultra-compact HTML table formatting for comparison results with minimal column widths.
   
6. `plot_dataset_comparisons()`: Plots comparisons of key metrics (e.g., total time, BEST scores, n_estimators) between experiment groups.


Additional Utilities:
---------------------
- `bootstrap_effect_size_alternative()`: Computes effect sizes with bootstrap confidence intervals
- `parse_log_tail()`: Parses experiment log files to extract best configurations
- `declared_params()`: Inspects function signatures for parameter information

Requires:
---------
- "dill>=0.3.8"
- "typing_extensions>=3.10" 
- "matplotlib>=3.5"
- "seaborn>=0.11.2"
- "scipy>=1.7.0"
- "numpy>=1.20.0"
- "pandas>=1.3.0"
- Dependencies from `rf_plateau_hpo` library (specified in pyproject.toml)

Note:
-----
This module is designed to work with the `rf_plateau_hpo` library and expects specific
data structures and function signatures from that library.
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


# Main experiment-runner function 
def run_experiment(
    method: Callable,
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    outdir: Union[str, Path] = "",
    max_features_grid: Sequence[object] = ("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0),
    max_depth_range: Tuple[int, int] = (1, 40),
    min_samples_leaf_range: Tuple[int, int] = (1, 20),
    min_samples_split_range: Tuple[int, int] = (2, 40),
    tune_criterion: bool = True,
    criterion: Optional[str] = None,
    class_weight: Optional[Union[str, Dict[str, float], List[Dict[str, float]]]] = None,
    n_trials: int = 40,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: int = 0,
    n_estimators_range: Optional[Tuple[int, int]] = (50, 2000),
    n_estimators_start: Optional[int] = 100,
    scale_factor: Optional[float] = 1.5,
    delta: Optional[float] = 1e-3,
    max_trees: Optional[int] = 10000,
    dataset: str = "",
    note: str = ""
) -> Tuple[Dict[str, Any], Path, Path]:
    """
    Run experiment with specified tuning method and save results.
    
    Returns:
        Tuple of (run_parameters, log_file_path, dill_file_path)
    """
    # Setup paths
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now(timezone.utc)
    random_state_ = int(ts.timestamp()) if random_state is None else random_state
    
    run_id = '-'.join([method.__name__, dataset, str(n_trials), str(random_state_)])
    log_file = outdir / f"{run_id}.log"
    dill_file = outdir / f"{run_id}.dill"
    
    # Prepare all parameters
    params = {
        'X': X, 'y': y, 'problem': problem, 'score_func': score_func,
        'greater_is_better': greater_is_better, 'max_features_grid': max_features_grid,
        'max_depth_range': max_depth_range, 'min_samples_leaf_range': min_samples_leaf_range,
        'min_samples_split_range': min_samples_split_range, 'tune_criterion': tune_criterion,
        'criterion': criterion, 'class_weight': class_weight, 'n_trials': n_trials,
        'n_jobs': n_jobs, 'random_state': random_state_, 'verbose': verbose,
        'log_file': log_file, 'refit': False,
        'n_estimators_range': n_estimators_range, 'n_estimators_start': n_estimators_start,
        'scale_factor': scale_factor, 'delta': delta, 'max_trees': max_trees
    }
    
    # Remove method-specific unused parameters
    if method == tune_rf_oob:
        plateau_params = TUNE_RF_OOB_PLATEAU_PARAMS_DFLT.keys() - TUNE_RF_OOB_PARAMS_DFLT.keys()
        params = {k: v for k, v in params.items() if k not in plateau_params}
        _, study = tune_rf_oob(**params)
        output = {'study': study}
    else:
        oob_params = TUNE_RF_OOB_PARAMS_DFLT.keys() - TUNE_RF_OOB_PLATEAU_PARAMS_DFLT.keys()
        params = {k: v for k, v in params.items() if k not in oob_params}
        _, best_n_estimators, study, plateau_found = tune_rf_oob_plateau(**params)
        output = {'best_n_estimators': best_n_estimators, 'study': study, 'plateau_found': plateau_found}
    
    # Parse log and prepare results
    log_parsed = parse_log_tail(log_file)
    
    # Prepare dataset metadata
    data_params = {'dataset': dataset}
    if hasattr(X, 'shape'): data_params['X.shape'] = X.shape
    if hasattr(X, 'columns'): data_params['X.columns'] = X.columns
    if hasattr(y, 'name'): data_params['y.name'] = y.name
    
    # Remove large data objects before serialization
    del params['X'], params['y']
    
    run_params = {
        'note': note, 'method': method.__name__,
        'experiment_started': ts.strftime("%Y-%m-%d %H:%M:%S"),
        'params_data': data_params, 'params_in': params,
        'params_out': output, 'params_parsed': log_parsed,
    }
    
    # Save to dill file
    with open(dill_file, "wb") as f:
        dill.dump(run_params, f, protocol=dill.HIGHEST_PROTOCOL)
    
    if verbose > 0:
        print(f"Log file: {log_file}\nDill file: {dill_file}")
    
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
    delta_range: List[float] = [1e-3],
    n_trials_range: List[int] = [40, 120],
    n_experiments: int = 20,
    **run_experiment_args
):
    for tune_criterion in tune_criterion_range:
        for depth_trees_only in depth_trees_only_range:
            for method in method_range:
                deltas = delta_range if method == tune_rf_oob_plateau else [None]

                for delta in deltas:
                    for n_trials in n_trials_range:
                        outdir = Path(dataset) / f"tune_criterion={tune_criterion}" / f"depth_trees_only={depth_trees_only}" / method.__name__
                        if delta is not None:
                            outdir = outdir / f"delta={delta:.0e}".replace('e-0', 'e-')
                        outdir = outdir / f"n_trials={n_trials}"

                        kwargs = run_experiment_args.copy()
                        kwargs.update({
                            'tune_criterion': tune_criterion,
                            'delta': delta,  # Safe for both methods
                        })

                        if depth_trees_only:
                            kwargs.update({
                                'min_samples_leaf_range': (1, 1),
                                'min_samples_split_range': (2, 2),
                                'max_features_grid': ('sqrt',),
                            })

                        for K in range(n_experiments):
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

def read_experiment_results(
    folder_path: str, 
    save_plots: bool = True, 
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Analyze experiment results from .dill files in a directory.
    
    Args:
        folder_path: Path to directory containing .dill experiment files
        save_plots: Whether to save generated plots to disk
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary containing extracted metrics and computed statistics
    """
    
    def generate_plots(results: Dict[str, Any]) -> None:
        """Generate visualizations for experiment metrics."""
        def save_show_plot(filename):
            if save_plots:
                plt.savefig(Path(folder_path) / filename, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()
        
        # Boxplots for individual metrics
        for data, title, filename in [
            (results['BEST scores'], "BEST Scores", "best_scores.jpg"),
            (results['Total time'], "Total Time", "total_time.jpg"), 
            (results['n_estimators'], "BEST n_estimators", "n_estimators.jpg")
        ]:
            if data:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=data)
                plt.scatter(np.ones(len(data)), data, color='red', alpha=0.5)
                plt.title(title)
                save_show_plot(filename)
        
        # Scatter plot for n_estimators vs max_depth
        if (results['n_estimators'] and results['max_depth'] and
            len(results['n_estimators']) == len(results['max_depth'])):
            
            plt.figure(figsize=(10, 6))
            plt.scatter(results['n_estimators'], results['max_depth'], c='blue', alpha=0.5)
            plt.title("n_estimators vs max_depth")
            plt.xlabel("n_estimators")
            plt.ylabel("max_depth")
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            save_show_plot("n_estimators_vs_max_depth.jpg")
    
    # Initialize results structure
    results = {
        'BEST scores': [], 'Total time': [], 'n_estimators': [], 'max_depth': [],
        'params_data': None, 'problem': None, 'greater_is_better': None,
        'BEST scores mean': None, 'BEST scores std': None,
        'Total time mean': None, 'Total time std': None,
        'n_estimators mean': None, 'n_estimators std': None,
    }
    
    # Load and process experiment files
    dill_files = list(Path(folder_path).glob("*.dill"))
    if not dill_files:
        return results
    
    for dill_file in dill_files:
        try:
            with open(dill_file, 'rb') as f:
                experiment = dill.load(f)
            
            # Extract metrics from experiment data
            params_parsed = experiment.get('params_parsed', {})
            params_in = experiment.get('params_in', {})
            params_out = experiment.get('params_out', {})
            
            best_score = params_parsed.get('BEST score')
            total_time = params_parsed.get('Total time')
            max_depth = params_parsed.get('BEST params', {}).get('max_depth')
            
            # Extract n_estimators from appropriate source
            n_estimators = (
                params_out.get('best_n_estimators') or  # Plateau method
                params_parsed.get('BEST params', {}).get('n_estimators')  # Standard method
            )
            
            # Validate required metrics
            if all(v is not None for v in [best_score, total_time, n_estimators, max_depth]):
                results['BEST scores'].append(best_score)
                results['Total time'].append(total_time)
                results['n_estimators'].append(n_estimators)
                results['max_depth'].append(max_depth)
                
                # Set metadata from first valid experiment
                if results['params_data'] is None:
                    results['params_data'] = experiment.get('params_data')
                    results['problem'] = params_in.get('problem')
                    results['greater_is_better'] = params_in.get('greater_is_better')
                    
        except (KeyError, IOError, dill.UnpicklingError) as e:
            print(f"Error processing {dill_file}: {e}")
            continue
    
    # Compute statistics for collected metrics
    def compute_stats(values, prefix):
        if values:
            array_vals = np.array(values)
            results[f'{prefix} mean'] = np.mean(array_vals)
            results[f'{prefix} std'] = np.std(array_vals)
    
    compute_stats(results['BEST scores'], 'BEST scores')
    compute_stats(results['Total time'], 'Total time')
    compute_stats(results['n_estimators'], 'n_estimators')
    
    # Generate visualizations
    if any(results[key] for key in ['BEST scores', 'Total time', 'n_estimators']):
        generate_plots(results)
    
    return results


def bootstrap_effect_size_alternative(
    sample1: np.ndarray, 
    sample2: np.ndarray, 
    n_bootstrap: int = 10000, 
    alternative: Literal['two-sided', 'less', "greater"] = 'two-sided'
) -> Tuple[str, float, float, float]:
    """
    Alternative version using Cliff's Delta for non-normal data.
    """
    if len(sample1) == 0 or len(sample2) == 0:
        raise ValueError("Input samples cannot be empty")

    # Normality check to choose effect size measure
    _, p_norm1 = stats.shapiro(sample1)
    _, p_norm2 = stats.shapiro(sample2)
    is_normal = p_norm1 > 0.05 and p_norm2 > 0.05
    n1, n2 = len(sample1), len(sample2)

    if is_normal:
        test_stat, test_p = stats.ttest_ind(
            sample1, sample2, equal_var=False, alternative=alternative
        )
        test_name = "t-test"
    else:
        test_stat, test_p = stats.mannwhitneyu(
            sample1, sample2, alternative=alternative
        )
        test_name = "MW"   

    def calculate_cohens_d(bs1, bs2):
        mean_diff = np.mean(bs1) - np.mean(bs2)
        var1 = np.var(bs1, ddof=1)
        var2 = np.var(bs2, ddof=1)
        n1_bs, n2_bs = len(bs1), len(bs2) 
        pooled_var = ((n1_bs - 1) * var1 + (n2_bs - 1) * var2) / (n1_bs + n2_bs - 2)
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 1e-10

        return mean_diff / pooled_std

    def calculate_cliffs_delta(bs1, bs2):
        """Calculate Cliff's Delta for non-parametric effect size"""
        n1_bs, n2_bs = len(bs1), len(bs2)
        
        # Count comparisons
        greater = np.sum(bs1[:, None] > bs2[None, :])
        less = np.sum(bs1[:, None] < bs2[None, :])
        
        return (greater - less) / (n1_bs * n2_bs)

    # Choose effect size calculator
    effect_calculator = calculate_cohens_d if is_normal else calculate_cliffs_delta

    # Generate bootstrap distribution
    effects = []
    for _ in range(n_bootstrap):
        bs1 = np.random.choice(sample1, size=n1, replace=True)
        bs2 = np.random.choice(sample2, size=n2, replace=True)
        effects.append(effect_calculator(bs1, bs2))
    
    effects = np.array(effects)
    
    # Calculate effect p-value from bootstrap distribution
    # i.e. the probability of zero effect 
    cdf_at_0 = np.mean(effects <= 0)
    
    if alternative == 'two-sided':
        effect_p_value = 2 * min(cdf_at_0, 1 - cdf_at_0)
    elif alternative == 'less':
        effect_p_value = 1 - cdf_at_0
    elif alternative == "greater":
        effect_p_value = cdf_at_0
    else:
        raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'")
    
    effect_mean = np.mean(effects)
    
    return test_name, test_p, effect_mean, effect_p_value

_BASE = "BASE"
_PLAT = "PLAT"


def compare_experiment_groups(
    dataset_folder: str, 
    delta_dir: str = "delta=1e-3",
    n_trials: Tuple[int, int] = (40, 120), 
    save_plots: bool = True
) -> pd.DataFrame:
    """
    Compare experiment groups using statistical tests and effect size measures.
    
    Performs comprehensive comparisons between different experimental configurations
    including algorithm variants, hyperparameter settings, and trial counts.
    
    Args:
        dataset_folder: Path to the dataset folder containing experiment results
        delta_dir: Delta parameter directory for plateau method.
        n_trials: Tuple of (low, high) trial counts to compare
        save_plots: Whether to save generated plots during analysis
        
    Returns:
        DataFrame with comparison results using MultiIndex columns
    """
    # Constants for column names
    TUNE_CRITERION = "tune criterion"
    DEPTH_ONLY = "only depth"
    ALGORITHM = "algorithm"
    N_TRIALS = f"$n_{{trials}}$"
    SIGNIFICANCE = "t-test $p_{value}$, Cohen's $d$ (or Mann-Whitney $p_{value}$, Cliff's $\delta$)"
    
    def yn(x: bool) -> str:
        return "YES" if x else "NO"
    
    def check(
        path1: str, path2: str, param: str, *, 
        alternative: str = "two-sided", save_plots: bool = True
    ) -> Tuple[str, Optional[Dict], Optional[Dict]]:
        """
        Compare two experiment paths for a specific parameter.
        """
        exp_res1 = read_experiment_results(path1, save_plots=save_plots)
        exp_res2 = read_experiment_results(path2, save_plots=save_plots)
        
        # Validate data availability
        if exp_res1[param] is None or exp_res2[param] is None:
            return "", exp_res1, exp_res2
        try:
            if len(exp_res1[param]) == 0 or len(exp_res2[param]) == 0:
                return "", exp_res1, exp_res2
        except (TypeError, AttributeError):
            return "", exp_res1, exp_res2
        
        # Handle score direction for BEST scores
        current_alternative = alternative
        if param == "BEST scores":
            gib1 = exp_res1.get("greater_is_better")
            gib2 = exp_res2.get("greater_is_better")
            
            if gib1 is None or gib2 is None:
                return "", exp_res1, exp_res2
            if gib1 != gib2:
                raise ValueError("greater_is_better must be equivalent in both experiments")
            
            # Flip alternative for minimization scores
            if not gib1 and alternative != "two-sided":
                current_alternative = "less" if alternative == "greater" else "greater"
        
        # Perform statistical comparison
        test_name, test_p, effect_mean, effect_p_value = bootstrap_effect_size_alternative(
            exp_res1[param], exp_res2[param], alternative=current_alternative
        )
        
        return _format_comparison_result(test_name, test_p, effect_mean), exp_res1, exp_res2
    
    def _format_comparison_result(test_name: str, test_p: float, effect_mean: float) -> str:
        """Format statistical comparison results for display."""
        d_name = r"\delta" if "MW" in test_name else "d"
        effect_threshold = 0.5 if d_name == "d" else 0.28
        
        # Format p-value
        test_p_str = f"{test_p:.1e}"
        if 'e-' in test_p_str:
            base, exp = test_p_str.split('e-')
            test_p_str = f"{base}e-{exp.lstrip('0')}"
        
        # Format effect size
        effect_mean_str = f"{effect_mean:.2f}"
        
        # Apply bold formatting for significant results
        p_value_formatted = (
            r"$\mathbf{" + test_p_str + "}$" if test_p < 0.05 else f"${test_p_str}$"
        )
        effect_formatted = (
            f"${d_name}=\\mathbf{{{effect_mean_str}}}$" 
            if abs(effect_mean) >= effect_threshold 
            else f"${d_name}={effect_mean_str}$"
        )
        
        return f"{p_value_formatted},\\\\ {effect_formatted}"
    
    def _build_comparison_row(
        tune_criterion: Union[bool, str], depth_trees_only: Union[bool, str], 
        algorithm: str, epsilon: str, n_trials_display: str, significance: str
    ) -> Dict:
        """Build a standardized comparison row."""
        return {
            TUNE_CRITERION: tune_criterion if isinstance(tune_criterion, str) else yn(tune_criterion),
            DEPTH_ONLY: depth_trees_only if isinstance(depth_trees_only, str) else yn(depth_trees_only),
            ALGORITHM: algorithm,
            r'$\varepsilon$': epsilon,
            N_TRIALS: n_trials_display,
            SIGNIFICANCE: significance
        }
    
    def _get_dataset_display_name(exp_res: Optional[Dict]) -> str:
        """Extract formatted dataset name from experiment results."""
        if exp_res is None:
            return os.path.basename(Path(dataset_folder)).replace("_", " ")
        
        params_data = exp_res.get('params_data', {})
        problem = exp_res.get('problem')
        
        if problem and params_data:
            dataset = params_data.get('dataset', '')
            shape = params_data.get('X.shape', ('', ''))
            n, p = shape if hasattr(shape, '__len__') and len(shape) == 2 else ('', '')
            if dataset != '' and n != '' and p != '': 
                return f"{dataset.replace('_', ' ')}, {n}x{p}, {problem}"
        
        return os.path.basename(Path(dataset_folder)).replace("_", " ")
    
    # Define all comparison configurations
    comparison_configs = []
    
    # Comparison 1: BASE algorithm with different n_trials
    for tune in [False, True]:
        for depth in [True, False]:
            comparison_configs.append({
                'path1': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob' / f'n_trials={n_trials[1]}',
                'path2': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob' / f'n_trials={n_trials[0]}',
                'param': "BEST scores", 'alternative': "greater",
                'tune': tune, 'depth': depth, 'algo': _BASE, 'epsilon': "",
                'n_trials_display': f"Score: {n_trials[1]} vs {n_trials[0]}"
            })
    
    # Comparison 2: PLAT algorithm with different n_trials  
    for tune in [False, True]:
        for depth in [True, False]:
            comparison_configs.append({
                'path1': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob_plateau' / delta_dir / f'n_trials={n_trials[1]}',
                'path2': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob_plateau' / delta_dir / f'n_trials={n_trials[0]}',
                'param': "BEST scores", 'alternative': "greater",
                'tune': tune, 'depth': depth, 'algo': _PLAT, 'epsilon': delta_dir.split('=')[1],
                'n_trials_display': f"Score: {n_trials[1]} vs {n_trials[0]}"
            })
    
    # Comparison 3: BASE vs PLAT algorithm comparison
    for tune in [False, True]:
        for depth in [True, False]:
            comparison_configs.append({
                'path1': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob' / f"n_trials={n_trials[1]}",
                'path2': Path(dataset_folder) / f'tune_criterion={tune}' / f'depth_trees_only={depth}' / 'tune_rf_oob_plateau' / delta_dir / f"n_trials={n_trials[1]}",
                'param': "BEST scores", 'alternative': "two-sided",
                'tune': tune, 'depth': depth, 'algo': f"Score: {_BASE} vs {_PLAT}", 
                'epsilon': delta_dir.split('=')[1], 'n_trials_display': n_trials[1]
            })

    # Comparison 4: Depth tuning comparison 
    for algo in [_BASE, _PLAT]:
        for tune in [False, True]:
            alg_path = 'tune_rf_oob' if algo == _BASE else Path('tune_rf_oob_plateau') / delta_dir
            comparison_configs.append({
                'path1': Path(dataset_folder) / f'tune_criterion={tune}' / 'depth_trees_only=False' / alg_path / f"n_trials={n_trials[1]}",
                'path2': Path(dataset_folder) / f'tune_criterion={tune}' / 'depth_trees_only=True' / alg_path / f"n_trials={n_trials[1]}",
                'param': "BEST scores", 'alternative': "two-sided",
                'tune': tune, 'depth': f"Score: {yn(False)} vs {yn(True)}", 'algo': algo,
                'epsilon': "" if algo == _BASE else delta_dir.split('=')[1], 'n_trials_display': n_trials[1]
            })

    # Comparison 5: Criterion tuning comparison
    for algo in [_BASE, _PLAT]:
        for depth in [False, True]:
            alg_path = 'tune_rf_oob' if algo == _BASE else Path('tune_rf_oob_plateau') / delta_dir
            comparison_configs.append({
                'path1': Path(dataset_folder) / 'tune_criterion=True' / f'depth_trees_only={depth}' / alg_path / f"n_trials={n_trials[1]}",
                'path2': Path(dataset_folder) / 'tune_criterion=False' / f'depth_trees_only={depth}' / alg_path / f"n_trials={n_trials[1]}",
                'param': "BEST scores", 'alternative': "two-sided",
                'tune': f"Score: {yn(True)} vs {yn(False)}", 'depth': depth, 'algo': algo,
                'epsilon': "" if algo == _BASE else delta_dir.split('=')[1], 'n_trials_display': n_trials[1]
            })

    # Comparison 6 and 7: BASE vs PLAT by time and by n_estimators
    for param in ['Total time', 'n_estimators']:
        algo_display = "Time" if param == 'Total time' else "$T$"
        comparison_configs.append({
            'path1': Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob' / f"n_trials={n_trials[1]}",
            'path2': Path(dataset_folder) / 'tune_criterion=False' / 'depth_trees_only=False' / 'tune_rf_oob_plateau' / delta_dir / f"n_trials={n_trials[1]}",
            'param': param, 'alternative': "two-sided",
            'tune': False, 'depth': False, 'algo': f"{algo_display}: {_BASE} vs {_PLAT}",
            'epsilon': delta_dir.split('=')[1], 'n_trials_display': n_trials[1]
        })
    
    # Execute all comparisons
    results = []
    exp_res_reference = None
    
    for config in comparison_configs:
        significance, er1, er2 = check(
            config['path1'], config['path2'], config['param'], 
            alternative=config['alternative'], save_plots=save_plots
        )
        exp_res_reference = er1 or er2 or exp_res_reference
        
        results.append(_build_comparison_row(
            config['tune'], config['depth'], config['algo'], 
            config['epsilon'], config['n_trials_display'], significance
        ))
    
    # Convert results to MultiIndex DataFrame
    dataset_display_name = _get_dataset_display_name(exp_res_reference)
    
    formatted_results = []
    for result in results:
        formatted_result = {
            (SIGNIFICANCE, dataset_display_name) if k == SIGNIFICANCE else ('', k): v 
            for k, v in result.items()
        }
        formatted_results.append(formatted_result)
    
    if formatted_results:
        columns = pd.MultiIndex.from_tuples(formatted_results[0].keys(), names=['', ''])
        return pd.DataFrame(formatted_results, columns=columns)
    
    return pd.DataFrame()


def process_html_table(html_text, padding_horizontal=3, padding_vertical=1, font_size=0.8, max_col_width=90):
    """
    Ultra-compact HTML table with absolutely minimal column widths
    
    Args:
        html_text: HTML table code
        padding_horizontal: horizontal padding in pixels
        padding_vertical: vertical padding in pixels
        font_size: font size in em units
        max_col_width: maximum column width in pixels

    Usage:
        df = compare_experiment_groups(dataset)
        html = df.to_html(notebook=True, escape=False, index=False, classes='dataframe')
        display(HTML(process_html_table(html)))
    """
    # Replace line breaks
    html_text = html_text.replace(r":", ":<br>")
    html_text = html_text.replace(r"\\", "<br>")
    
    # Process cell content formatting
    def reformat_cells(html):
        # First replace \mathbf{...} with <b>...</b> in both parts
        def replace_mathbf(text):
            # Pattern to match \mathbf{content}
            pattern = r'\\mathbf{([^}]*)}'
            return re.sub(pattern, r'<b>\1</b>', text)

        # Pattern to match the cell structure: $...$,<br> $...=...$
        pattern = r'(\$[^$]+\$),\s*<br>\s*(\$[^=]+)=([^$]+\$)'

        def reformat_match(match):
            first_part = replace_mathbf(match.group(1).strip('$'))
            variable = replace_mathbf(match.group(2).strip('$'))
            value = replace_mathbf(match.group(3).strip('$'))

            # Remove any remaining $ signs and apply formatting
            first_part = first_part.replace('$', '')
            variable = variable.replace('$', '')
            value = value.replace('$', '')

            return f'{first_part},<br> ${variable}$={value}'

        # Apply the main transformation
        html = re.sub(pattern, reformat_match, html)

        # Additional pass to handle any remaining \mathbf commands
        html = replace_mathbf(html)
    
        return html

    html_text = reformat_cells(html_text)
    
    # CSS for ultra-minimal column widths
    css = f"""
    <style>
    .dataframe {{
        border-collapse: collapse;
        border-spacing: 0;
        font-size: {font_size}em;
        width: auto !important;
        table-layout: auto; /* Let columns shrink to content */
        margin: 0;
        padding: 0;
    }}
    .dataframe td, .dataframe th {{
        padding: {padding_vertical}px {padding_horizontal}px !important;
        border: 1px solid #ddd;
        text-align: center;
        vertical-align: middle;
        line-height: 1;
        max-width: {max_col_width}px;
        min-width: 0; /* Allow columns to shrink below content width if needed */
        width: auto; /* Let browser determine optimal width */
        white-space: nowrap; /* Critical: prevent text wrapping */
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .dataframe th {{
        background-color: #f5f5f5;
        font-weight: bold;
        white-space: normal; /* Only headers can wrap */
        word-break: break-word;
        hyphens: auto;
    }}
    /* Handle cells with explicit line breaks */
    .dataframe td:has(br) {{
        white-space: normal;
        line-height: 1.2;
    }}
    .dataframe td br {{
        display: block;
        margin: 1px 0;
    }}
    /* Make numeric/content columns even more compact */
    .dataframe td:contains("."),
    .dataframe td:contains("e"),
    .dataframe td:contains("$") {{
        font-family: "Courier New", monospace; /* Monospace for consistent width */
        letter-spacing: -0.5px; /* Tighten spacing */
    }}
    </style>
    """
    
    return f'<div style="overflow-x: auto; display: block; width: fit-content;">{html_text}</div>' + css


def plot_dataset_comparisons(
    dataset_folders: List[str],
    tune_criterion: bool = True,
    depth_trees_only: bool = False,
    delta_dir: Union[str, dict] = "delta=1e-3",
    n_trials: int = 120,
    ncols: Optional[int] = None,
    show_scores: bool = False,
    bw: float = 0.9,
    min_w: float = 3.0,
    min_h: float = 2.5,
    tight_layout_top: float = 0.9,
    save_plots: bool = True
) -> None:
    """
    Create comparative visualizations of experiment results across multiple datasets.
    
    For each dataset, displays paired bar groups showing:
    - Total time (left y-axis, seconds)
    - Best scores (middle y-axis, score function units) - optional
    - n_estimators (right y-axis, integer ticks)
    
    Each metric shows two bars:
    - tune_rf_oob() (solid fill)
    - tune_rf_oob_plateau() (hatched pattern)
    
    Args:
        dataset_folders: List of dataset folder paths to analyze
        tune_criterion: Whether criterion tuning was enabled
        depth_trees_only: Whether only depth and trees were tuned
        delta_dir: Delta parameter directory or a dict like {'iris': '1e-3', 'wine': '1e-2'} for plateau method.
        n_trials: Number of trials to analyze
        ncols: Number of columns in subplot grid (auto-calculated if None)
        show_scores: Whether to display BEST scores boxplots
        bw: Bar width for spacing control
        min_w: Minimum width per subplot column in inches
        min_h: Minimum height per subplot row in inches
        tight_layout_top: Top margin for tight layout
        save_plots: Whether to save generated plots
        
    Returns:
        None
    """
    # Setup
    n_datasets = len(dataset_folders)
    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)
    
    # Calculate figure size with dynamic scaling
    fig_width = ncols * min_w
    fig_height = nrows * min_h + 1.0
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_list = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    # Visual styling configuration
    TITLE_FONT_SIZE = 12
    AXIS_LABEL_FONT_SIZE = 12  
    TICK_LABEL_FONT_SIZE = 11
    LEGEND_FONT_SIZE = 12
    
    # Color scheme for different metrics
    COLOR_TIME = "#4C72B0"      # Total time bars
    COLOR_N_ESTIMATORS = "#C44E52"  # n_estimators bars  
    COLOR_SCORES = "#55A868"    # BEST scores bars
    
    HATCH_PLATEAU = "//"        # Hatch pattern for plateau method
    
    legend_handles, legend_labels = [], []

    for idx, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[idx]
        
        delta_dir_ = delta_dir[dataset_folder] if isinstance(delta_dir, dict) else delta_dir

        # Load results
        base_path = Path(dataset_folder) / f"tune_criterion={tune_criterion}" / f"depth_trees_only={depth_trees_only}"
        oob_path = base_path / "tune_rf_oob" / f"n_trials={n_trials}"
        plateau_path = base_path / "tune_rf_oob_plateau" / delta_dir_ / f"n_trials={n_trials}"
        print(plateau_path, delta_dir_)
        
        try:
            res_oob = read_experiment_results(oob_path, save_plots=save_plots)
            res_plateau = read_experiment_results(plateau_path, save_plots=save_plots)
        except (FileNotFoundError, KeyError):
            continue
        
        # Validate data
        required_metrics = ["Total time mean", "n_estimators mean", "BEST scores"]
        if not all(res_oob.get(metric) for metric in required_metrics):
            continue
        if not all(res_plateau.get(metric) for metric in required_metrics):
            continue
        
        # Prepare data
        oob_data = {
            'time': (res_oob["Total time mean"], res_oob["Total time std"]),
            'n_est': (res_oob["n_estimators mean"], res_oob["n_estimators std"]),
            'scores': res_oob["BEST scores"]
        }
        plateau_data = {
            'time': (res_plateau["Total time mean"], res_plateau["Total time std"]),
            'n_est': (res_plateau["n_estimators mean"], res_plateau["n_estimators std"]),
            'scores': res_plateau["BEST scores"]
        }
        
        # Define positions
        if show_scores:
            x_time = [0.0, 1.0]
            x_scores = [2.5, 3.5]
            x_n_estimators = [5.0, 6.0]
        else:
            x_time = [0.0, 1.0]
            x_n_estimators = [3.0, 4.0]
        
        # Time bars (left axis)
        bars_time = ax.bar(
            x_time, 
            [oob_data['time'][0], plateau_data['time'][0]],
            yerr=[oob_data['time'][1], plateau_data['time'][1]],
            width=bw, color=COLOR_TIME, hatch=["", HATCH_PLATEAU],
            edgecolor="black", linewidth=0.8
        )
        ax.set_ylabel("Time (s)", fontsize=AXIS_LABEL_FONT_SIZE)
        
        # n_estimators bars (right axis)
        ax2 = ax.twinx()
        bars_n = ax2.bar(
            x_n_estimators,
            [oob_data['n_est'][0], plateau_data['n_est'][0]],
            yerr=[oob_data['n_est'][1], plateau_data['n_est'][1]],
            width=bw, color=COLOR_N_ESTIMATORS, hatch=["", HATCH_PLATEAU],
            edgecolor="black", linewidth=0.8
        )
        ax2.set_ylabel("$T$", fontsize=AXIS_LABEL_FONT_SIZE)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Scores boxplots (middle axis, optional)
        if show_scores:
            ax3 = ax.twinx()
            # Position scores axis between time and n_estimators
            # ax3.spines['right'].set_position(('axes', 1.1))
            box_scores = ax3.boxplot(
                [oob_data['scores'], plateau_data['scores']],
                positions=x_scores, widths=bw, patch_artist=True,
                boxprops=dict(facecolor=COLOR_SCORES, edgecolor="black", linewidth=0.8),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                flierprops=dict(markerfacecolor="red", marker="o", markersize=5),
                medianprops=dict(color="orange", linewidth=2.5)
            )
            box_scores['boxes'][1].set_hatch(HATCH_PLATEAU)
            ax3.yaxis.set_major_locator(plt.NullLocator())
        
        # Title and styling
        ax.set_title(Path(dataset_folder).name, fontsize=TITLE_FONT_SIZE)
        ax.set_xticks([])
        ax.grid(False)
        ax2.grid(False)
        if show_scores:
            ax3.grid(False)
        
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
        ax2.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
        
        # Build legend once (preserving order: time, scores, estimators)
        if not legend_handles:
            legend_items = [
                (bars_time[0], f"Time {_BASE}"),
                (bars_time[1], f"Time {_PLAT}")
            ]
            
            if show_scores:
                legend_items.extend([
                    (box_scores["boxes"][0], f"Score {_BASE}"),
                    (box_scores["boxes"][1], f"Score {_PLAT}")
                ])
            
            legend_items.extend([
                (bars_n[0], f"$T$ {_BASE}"),
                (bars_n[1], f"$T$ {_PLAT}")
            ])
            
            legend_handles, legend_labels = zip(*legend_items)
    
    # Cleanup and finalize
    for j in range(len(dataset_folders), len(axes_list)):
        fig.delaxes(axes_list[j])
    
    fig.legend(legend_handles, legend_labels, loc="upper center", 
               ncol=3 if show_scores else 2, fontsize=LEGEND_FONT_SIZE,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    
    fig.tight_layout(rect=[0, 0, 1, tight_layout_top])
    plt.show()