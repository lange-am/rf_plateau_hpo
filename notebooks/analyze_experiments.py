"""
analyze_experiments.py
----------------------

Analysis and visualization helpers for Random Forest HPO experiments generated
by ``notebooks/run_experiments.py``.

The module aggregates ``.dill`` files, extracts Optuna study metadata, performs
statistical comparisons, exports compact HTML/LaTeX tables, and creates figures
used in the accompanying paper.

Key functions
-------------
- ``read_experiment_results()``: load and aggregate metrics from experiment files.
- ``bootstrap_effect_size_alternative()``: compute Cohen's d or Cliff's delta
  with bootstrap stabilization.
- ``experiment_comparison_table()``: build comparison tables with t-tests or
  Mann-Whitney tests and signed effect sizes.
- ``process_html_table()`` / ``tab2tex()``: format result tables for inspection
  and manuscript export.
- ``plot_dataset_comparisons()``: grouped bars for runtime and selected tree count.
- ``plot_delta_boxplots()``: boxplots of best scores versus PLATEAU tolerance.
- ``plot_B_trajectories()``: trial-wise PLATEAU trajectories with empirical
  frequency heatmaps over nominal tree-count levels.

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

from run_experiments import (
    RF_HPO_ALGORITHMS,
    DEFAULT_N_TRIALS_GRID,
    DEFAULT_DELTA_GRID,
    DEFAULT_SCALE_FACTOR_GRID,
    parse_study,
    get_experiment_directory,
    build_ladder,
)

# ---------- ANALYZE & VISUALIZE EXPERIMENTS ----------

def read_experiment_results(
    folder_path: Union[Path, str], 
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
    
    def generate_plots(title: str, values: List[Union[int, float, bool]]) -> None:
        """Generate visualizations for experiment metrics."""
        def save_show_plot(filename):
            if save_plots:
                plt.savefig(Path(folder_path) / filename, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close()
        
        # Boxplots for individual metrics
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=values)
        plt.scatter(np.ones(len(values)), values, color='red', alpha=0.5)
        plt.title(title)
        save_show_plot(f"{title}.jpg")
        
    # Initialize results structure
    res = {
        'time_total': [],
        'n_trials_total': [],
        'n_trees_built': [],
        'n_trials_shift_left': [],
        'n_trials_stay': [],
        'n_trials_shift_right': [],
        'n_trials_pruned': [],
        'n_trials_pruned_incomplete_score': [],
        'n_trials_pruned_should_prune': [],
        'n_trials_pruned_no_plateau': [],
        'BEST_score': [],
        'BEST_params': [],
        'BEST_n_estimators': [],
        'BEST_stop_status': [],
        'BEST_plateau_found': [], 
        'problem': [], 
        'greater_is_better': [],
        'n': [],
        'p': [],
        'dataset': [],
        'scale_factor': [],
        'n_trials': [],
        'n_estimators_start': [],
        'max_trees': [],
    # 
        'trees_built': [],
        'B': [],
        'pruned': [],
        'shift_left': [],
        'stay': [],
        'shift_right': [],
    }
    
    # Load and process experiment files
    dill_files = list(Path(folder_path).glob("*.dill"))
    if not dill_files:
        return {}

    for dill_file in dill_files:
        try:
            with open(dill_file, 'rb') as f:
                exprm = dill.load(f)

            params_data = exprm['params_data']
            params_in = exprm['params_in']
            params_out = exprm['params_out']

            dataset = params_data['dataset']
            n, p = params_data['X.shape']
            scale_factor = params_in['scale_factor']
            n_trials = params_in['n_trials']
            n_estimators_start = params_in['n_estimators_start']
            max_trees = params_in['max_trees']

            for k in res.keys():
                if k in ['problem', 'greater_is_better']:
                    res[k] += [params_in.get(k)]
                elif k == 'n':
                    res[k] += [n]
                elif k == 'p':
                    res[k] += [p]
                elif k == 'dataset':
                    res[k] += [dataset]
                elif k == 'scale_factor':
                    res[k] += [scale_factor]
                elif k == 'n_trials':
                    res[k] += [n_trials]
                elif k == 'n_estimators_start':
                    res[k] += [n_estimators_start]
                elif k == 'max_trees':
                    res[k] += [max_trees]
                elif k == 'B':
                    study = params_out.get('study', {}).get('study')
                    res[k] += [parse_study(study).get('B') if study else None]
                else:
                    value = params_out.get(k) or params_out.get('study', {}).get(k)
                    res[k] += [0 if 'n_' in k and value is None else value]

            res['BEST_n_estimators'][-1] = res['BEST_n_estimators'][-1] or res['BEST_params'][-1].get('n_estimators')
        except (KeyError, IOError, dill.UnpicklingError) as e:
            print(f"Error processing {dill_file}: {e}")
            continue

    COMMOM_PARAMS = [
        'problem', 
        'greater_is_better', 
        'n', 'p', 'dataset', 
        'scale_factor', 
        'n_trials',
        'n_estimators_start',
        'max_trees',
    ]
    for k in COMMOM_PARAMS:
        if len(set(res[k])) > 1:
            raise ValueError(f"All experiments must have the same '{k}'!")
        elif None in res[k]:
            raise ValueError(f"All experiments must have valid '{k}'!")
        elif len(res[k]) == 0:
            res[k] = None
        else:
            res[k] = res[k][0]

    res_ = {}
    for prm in res['BEST_params']:
        for k, v in prm.items():
            res_.setdefault(f"BEST_params_{k}", []).append(v)
    res.update(res_)

    res_ = {}
    for k, v in res.items():
        if k not in (['BEST_params'] + COMMOM_PARAMS) or k == 'B':
            try:
                if k == 'BEST_params_max_features':
                    mf_sqrt = 1/np.sqrt(res['p']) if res['p'] else None
                    v_ = [mf_sqrt if x == 'sqrt' else x for x in v]
                else:
                    v_ = v
                if any(x is not None for x in v_):
                    if isinstance(v_[0], list):
                        max_len = max(len(row) for row in v_)
                        arr = np.empty((len(v_), max_len), dtype=type(v_[0][0]))
                        for i, row in enumerate(v_):
                            arr[i, :len(row)] = row
                            arr[i, len(row):] = row[-1]
                    else:
                        arr = np.array(v_)
                    res_[f'{k}_mean'] = np.nanmean(arr, axis=0)
                    res_[f'{k}_std'] = np.nanstd(arr, axis=0)
                    if arr.ndim == 1:
                        generate_plots(k, v_)
            except:
                continue
    res.update(res_)

    return res


def bootstrap_effect_size_alternative(
    sample1: np.ndarray, 
    sample2: np.ndarray, 
    n_bootstrap: int = 10000, 
    alternative: Literal['two-sided', 'less', "greater"] = 'two-sided'
) -> Tuple[str, float, float, float]:
    """
    Alternative version using Cliff's Delta for non-normal data.
    """
    # if len(sample1) == 0 or len(sample2) == 0:
    #     raise ValueError("Input samples cannot be empty")

    # if len(sample1) < 5 or len(sample2) < 5:
    #     warnings.warn(
    #         f"At least one of sample sizes is too small: {len(sample1)}, {len(sample2)}.", 
    #         RuntimeWarning
    #     )

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


def _alg(method: RF_HPO_ALGORITHMS):
    return str(method)


def experiment_comparison_table(
    dataset_folder: str, 
    scale_factor: Tuple[float, float] = (DEFAULT_SCALE_FACTOR_GRID[0], DEFAULT_SCALE_FACTOR_GRID[-1]),
    delta: Tuple[float, float] = (DEFAULT_DELTA_GRID[0], DEFAULT_DELTA_GRID[-1]),
    n_trials: Tuple[int, int] = (DEFAULT_N_TRIALS_GRID[0], DEFAULT_N_TRIALS_GRID[-1]),
    save_plots: bool = True,
    show_epsilon_column = True,
    show_effect_size = True
) -> pd.DataFrame:
    """
    Compare experiment groups using statistical tests and effect size measures.
    
    Performs comprehensive comparisons between different experimental configurations
    including algorithm variants, hyperparameter settings, and trial counts.
    
    Args:
        dataset_folder: Path to the dataset folder containing experiment results
        delta_dir: Delta parameter directory for plateau method 
            (shown as \varepsilon in the table to avoid confusion with Cliff's \delta).
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
    SIGNIFICANCE = r"t-test p-value, Cohen's $d$ (or Mann-Whitney p-value, Cliff's $\delta$)"
    if not show_effect_size:
        SIGNIFICANCE = "t-test or Mann-Whitney p-value"
    
    def yn(x: bool) -> str:
        return "YES" if x else "NO"
    
    def check(
        path1: Union[Path, str], 
        path2: Union[Path, str], 
        param: str, *, 
        alternative: str = "two-sided", 
        save_plots: bool = True,
    ) -> Tuple[str, Optional[Dict], Optional[Dict]]:
        """
        Compare two experiment paths for a specific parameter.
        """
        exp_res1 = read_experiment_results(path1, save_plots=save_plots)
        exp_res2 = read_experiment_results(path2, save_plots=save_plots)

        # Validate data availability
        if exp_res1.get(param) is None or exp_res2.get(param) is None:
            return "", exp_res1, exp_res2
        try:
            sample1 = [x for x in exp_res1[param] if x is not None and not np.isnan(x)]
            sample2 = [x for x in exp_res2[param] if x is not None and not np.isnan(x)]

            if len(sample1) < 5 or len(sample2) < 5 or len(set(sample1)) == 1 or len(set(sample2)) == 1:
                return "", exp_res1, exp_res2
        except (TypeError, AttributeError):
            return "", exp_res1, exp_res2
        
        # Handle score direction for BEST scores
        current_alternative = alternative
        if param == "BEST_score":
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
            sample1, sample2, alternative=current_alternative
        )
        
        if not show_effect_size:
            effect_mean = None

        return _format_comparison_result(test_name, test_p, effect_mean), exp_res1, exp_res2
    
    def _format_comparison_result(
        test_name: str, 
        test_p: float, 
        effect_mean: Optional[float] = None
    ) -> str:
        """Format statistical comparison results for display."""
        # Format p-value in scientific notation
        test_p_str = f"{test_p:.1e}"
        
        # Convert to LaTeX scientific notation
        if 'e-' in test_p_str:
            base, exp = test_p_str.split('e-')
            exp_clean = exp.lstrip('0') or '0'
            test_p_str = f"{base}\\times10^{{-{exp_clean}}}"
        elif 'e+' in test_p_str:
            base, exp = test_p_str.split('e+')
            exp_clean = exp.lstrip('0') or '0'
            test_p_str = f"{base}\\times10^{{{exp_clean}}}"
        
        # Apply bold formatting for significant results
        p_value_formatted = (rf"$\mathbf{{{test_p_str}}}$" 
                           if test_p < 0.05 else f"${test_p_str}$")
        
        # Format effect size if present
        if effect_mean is not None:
            d_name = r"\delta" if "MW" in test_name else "d"
            effect_threshold = 0.5 if d_name == "d" else 0.28
            
            effect_mean_str = f"{effect_mean:.2f}"
            effect_formatted = (f"${d_name}=\\mathbf{{{effect_mean_str}}}$" 
                              if abs(effect_mean) >= effect_threshold 
                              else f"${d_name}={effect_mean_str}$")
            
            return f"{p_value_formatted}, {effect_formatted}"
        
        return f"{p_value_formatted}"

    def _build_comparison_row(
        tune_criterion: Union[bool, str], depth_trees_only: Union[bool, str], 
        algorithm: str, epsilon: str, n_trials_display: str, significance: str
    ) -> Dict:
        """Build a standardized comparison row."""
        row = {
            TUNE_CRITERION: tune_criterion if isinstance(tune_criterion, str) else yn(tune_criterion),
            DEPTH_ONLY: depth_trees_only if isinstance(depth_trees_only, str) else yn(depth_trees_only),
            ALGORITHM: algorithm,
            r'$\varepsilon$': epsilon,
            N_TRIALS: n_trials_display,
            SIGNIFICANCE: significance
        }

        if not show_epsilon_column:
            del row[r'$\varepsilon$']
        return row
    
    def _get_dataset_display_name(exp_res: Optional[Dict]) -> str:
        """Extract formatted dataset name from experiment results."""
        if exp_res is None:
            return os.path.basename(Path(dataset_folder)).replace("_", " ")
        
        problem = str(exp_res.get('problem'))
        n = int(exp_res.get('n'))
        p = int(exp_res.get('p'))
        dataset = str(exp_res.get('dataset'))
        
        if problem and n and p and dataset:
            return f"{dataset.replace('_', ' ')}, {n}x{p}, {problem}"
        
        return os.path.basename(Path(dataset_folder)).replace("_", " ")
    
    def _get_dir(
        tune_criterion: bool,
        depth_trees_only: bool,
        method: RF_HPO_ALGORITHMS,
        *,
        scale_factor: float = scale_factor[0],
        delta: float = delta[0],
        n_trials: int = n_trials[-1],
    ) -> Path:
        return get_experiment_directory(
            dataset_folder, tune_criterion, depth_trees_only,
            method=method, 
            scale_factor=scale_factor,
            delta=delta,
            n_trials=n_trials
        )

    # Define all comparison configurations
    conf_dflt = {
        'scale_factor': str(scale_factor[0]),
        'epsilon': str(delta[0]),
        'alternative': "two-sided",
        'n_trials_display': str(n_trials[1]),
    }
    comparison_configs = []
    
    # Comparison 1: TPE algorithm with different n_trials
    for tune in [True, False]:
        for depth in [True, False]:
            comparison_configs.append({**conf_dflt, 
                'path1': _get_dir(tune, depth, 'TPE', n_trials=n_trials[1]), 
                'path2': _get_dir(tune, depth, 'TPE', n_trials=n_trials[0]), 
                'param': "BEST_score", 'tune': tune, 'depth': depth, 'algo': _alg('TPE'),
                'alternative': "greater", 
                'n_trials_display': f"Score: {n_trials[1]} vs {n_trials[0]}",
                'epsilon': "",
            })
    
    # Comparison 2: PLATEAU algorithm with different n_trials  
    for tune in [True, False]:
        for depth in [True, False]:
            comparison_configs.append({**conf_dflt,
                'path1': _get_dir(tune, depth, 'PLATEAU', n_trials=n_trials[1]), 
                'path2': _get_dir(tune, depth, 'PLATEAU', n_trials=n_trials[0]), 
                'param': "BEST_score", 'tune': tune, 'depth': depth, 'algo': _alg('PLATEAU'),
                'alternative': "greater", 
                'n_trials_display': f"Score: {n_trials[1]} vs {n_trials[0]}",
            })
    
    # Comparison 3: TPE vs PLATEAU algorithm comparison
    for tune in [True, False]:
        for depth in [True, False]:
            comparison_configs.append({**conf_dflt,
                'path1': _get_dir(tune, depth, 'TPE'), 
                'path2': _get_dir(tune, depth, 'PLATEAU'), 
                'param': "BEST_score", 'tune': tune, 'depth': depth, 
                'algo': f"Score: {_alg('TPE')} vs {_alg('PLATEAU')}", 
            })

    # Comparison 4: Criterion tuning comparison
    for algo in ["TPE", "PLATEAU"]:
        for depth in [True, False]:
            comparison_configs.append({**conf_dflt,
                'path1': _get_dir(True, depth, algo), 
                'path2': _get_dir(False, depth, algo), 
                'param': "BEST_score", 'tune': f"Score: {yn(True)} vs {yn(False)}", 
                'depth': depth, 'algo': _alg(algo),
                'epsilon': "" if algo == "TPE" else str(delta[0]),
            })

    # Comparison 5: Depth tuning comparison 
    for algo in ["TPE", "PLATEAU"]:
        for tune in [True, False]:
            comparison_configs.append({**conf_dflt,
                'path1': _get_dir(tune, True, algo), 
                'path2': _get_dir(tune, False, algo), 
                'param': "BEST_score", 'tune': tune, 'depth': f"Score: {yn(True)} vs {yn(False)}", 
                'algo': _alg(algo), 'epsilon': "" if algo == "TPE" else str(delta[0]),
            })

    # Comparison 6: Fixed T vs dynamic T
    for algs in [("TPE_Tmin-Tmax", "TPE"), ("TPE_Tmin-PLT", "PLATEAU")]:
        comparison_configs.append({**conf_dflt,
            'path1': _get_dir(False, False, algs[0]), 
            'path2': _get_dir(False, False, algs[1]),
            'param': "BEST_score", 'tune': False, 'depth': False, 
            'algo': f"Score: {_alg(algs[0])} vs {_alg(algs[1])}",
        })

    # Comparison 7: Different 'n_trials_pruned'
    comparison_configs.append({**conf_dflt,
        'path1': _get_dir(False, False, "HB"), 
        'path2': _get_dir(False, False, "PLATEAU"), 
        'param': 'n_trials_pruned', 'tune': False, 'depth': False, 
        'algo': f"n_trials pruned: {_alg('HB')} vs {_alg('PLATEAU')}",
    })

    param_display = {
        'n_trees_built': 'trees built',
        'time_total': 'time' 
    }

    # Comparison 8: Different expencies
    for algs in [("HB", "TPE")]:
        for param in ['time_total', 'n_trees_built']:
            comparison_configs.append({**conf_dflt,
                'path1': _get_dir(False, False, algs[0]), 
                'path2': _get_dir(False, False, algs[1]), 
                'param': param, 'tune': False, 'depth': False, 
                'algo': f"{param_display[param]}: {_alg(algs[0])} vs {_alg(algs[1])}",
            })

    # Comparison 9: PLATEAU expencies at different scale_factor
    for param in ['time_total', 'n_trees_built']:
        comparison_configs.append({**conf_dflt,
            'path1': _get_dir(False, False, "PLATEAU", scale_factor=scale_factor[0]), 
            'path2': _get_dir(False, False, "PLATEAU", scale_factor=scale_factor[-1]),
            'param': param, 'tune': False, 'depth': False, 
            'algo': f"{param_display[param]}: sf={str(scale_factor[0])} vs sf={str(scale_factor[-1])}",
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
        columns = pd.MultiIndex.from_tuples(formatted_results[0].keys(), names=['level0', 'level1'])
        return pd.DataFrame(formatted_results, columns=columns)
    
    return pd.DataFrame()


def process_html_table(
    html_text, 
    padding_horizontal=3, 
    padding_vertical=1, 
    font_size=0.8, 
    max_col_width=90
):
    """
    Ultra-compact HTML table with absolutely minimal column widths
    
    Args:
        html_text: HTML table code
        padding_horizontal: horizontal padding in pixels
        padding_vertical: vertical padding in pixels
        font_size: font size in em units
        max_col_width: maximum column width in pixels

    Usage:
        df = experiment_comparison_table(dataset)
        html = df.to_html(index=False, classes='dataframe')
        display(HTML(process_html_table(html)))
    """
    # Replace colons with line breaks in headers
    html_text = html_text.replace(r":", ":<br>")
    
    # Process cell content formatting
    def reformat_cells(html):
        """Reformat LaTeX math to HTML-compatible format."""
        # Helper functions
        def replace_mathbf(text):
            """Replace \mathbf{...} with <b>...</b>"""
            return re.sub(r"\\mathbf{(.*?)}", r'<b>\1</b>', text)

        def convert_latex_scientific(text):
            """Convert \times10^{...} to e-notation"""
            def replace_exp(match):
                exp = match.group(1).replace('{', '').replace('}', '')
                return f"e{exp.lstrip('0') or '0'}" if exp.startswith('-') else f"e+{exp.lstrip('0').lstrip('+') or '0'}"

            return re.sub(r'\\times10\^{(.*)}', replace_exp, text)

        # Pattern to match p-value and effect size
        pattern_two_parts = r'\$([^$]+\\times[^$]+)\$,\s*\$([^=]+)=([^$]+)\$'
        # Pattern to match only p-value
        pattern_single_part = r'\$([^$]+\\times[^$]+)\$'

        # Process two parts (p-value and effect size)
        def reformat_two_parts(match):
            p_val, var, val = match.groups()
            p_val_processed = convert_latex_scientific(replace_mathbf(p_val))
            var_processed = replace_mathbf(var)
            val_processed = replace_mathbf(val)
            return f'{p_val_processed},<br>${var_processed}$={val_processed}'

        # Process single part (only p-value)
        def reformat_single_part(match):
            part = convert_latex_scientific(replace_mathbf(match.group(1)))
            return part

        # Apply transformations
        html_ = re.sub(pattern_two_parts, reformat_two_parts, html)
        if html_ == html:
            html_ = re.sub(pattern_single_part, reformat_single_part, html_)

        return html_    

    html_text = reformat_cells(html_text)
    
    # CSS for ultra-minimal column widths
    css = f"""
    <style>
    .dataframe {{
        border-collapse: collapse;
        border-spacing: 0;
        font-size: {font_size}em;
        width: auto !important;
        table-layout: auto;
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
        min-width: 0;
        width: auto;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .dataframe th {{
        background-color: #f5f5f5;
        font-weight: bold;
        white-space: normal;
        word-break: break-word;
        hyphens: auto;
    }}
    .dataframe td:has(br) {{
        white-space: normal;
        line-height: 1.2;
    }}
    .dataframe td br {{
        display: block;
        margin: 1px 0;
    }}
    .dataframe td:contains("."),
    .dataframe td:contains("e"),
    .dataframe td:contains("$") {{
        font-family: "Courier New", monospace;
        letter-spacing: -0.5px;
    }}
    </style>
    """
    
    return f'<div style="overflow-x: auto; display: block; width: fit-content;">{html_text}</div>' + css


def tab2tex(
    df_tex: pd.DataFrame, 
    halign: str = 'c',
    valign: str = 'm{1.6cm}',
    makecell_option = '*',
) -> str:
    """
    Convert DataFrame to LaTeX tabular with fixed-width columns.
    
    Args:
        df_tex: DataFrame from experiment_comparison_table()
        column_width: Width for LaTeX p{} columns
        align: Alignment inside p{} columns ('c', 'l', 'r')
    
    Returns:
        LaTeX tabular code
    """
    def bold_header(cols):
        def bold_cell(col_cell: str) -> str:
            if r'\textbf{' in col_cell or r'\mathbf{' in col_cell:
                return col_cell

            s = re.sub(r'\$([^\$]+)\$', lambda match: f'$\\mathbf{{{match.group(1)}}}$', col_cell)

            return f'\\textbf{{{s}}}' if s else ''

        if isinstance(cols, pd.MultiIndex):
            new_cols = []
            for level_idx in range(cols.nlevels):
                level_values = cols.get_level_values(level_idx)
                new_level = [bold_cell(str(val)) if pd.notna(val) else '' for val in level_values]
                new_cols.append(new_level)
            return pd.MultiIndex.from_arrays(new_cols)
        return [bold_cell(str(col)) if pd.notna(col) else '' for col in cols]

    # Build column specification with p{} and alignment
    alignment_cmd = {'c': '\\centering', 'l': '\\raggedright', 'r': '\\raggedleft'}.get(halign, '\\centering')
    col_spec = f">{{{alignment_cmd}\\arraybackslash}}{valign}"
    n_cols = len(df_tex.columns)
    col_format = '|'.join([''] + [col_spec]*n_cols + [''])
    
    # Process cell content - add line break for two-part results
    def process_cell(cell):
        cell_str = str(cell)
        cell_str = cell_str.replace(r": ", r":\\")
        pattern = r'\$(.*?)\$,\s*\$(.*?)\$'
        match = re.search(pattern, cell_str)

        cell_str = f"\\makecell{makecell_option}{{{cell_str}}}"
        if match:
            cell_str = f"\\makecell{makecell_option}{{${match.group(1)}$\\\\${match.group(2)}$}}"

        return cell_str
    
    # Apply processing and convert to LaTeX
    processed_df = df_tex.applymap(process_cell)
    processed_df.columns = bold_header(processed_df.columns)
    latex_str = processed_df.style.hide(axis="index").to_latex(
        column_format=col_format,
        multicol_align=halign+'|',
        hrules=True,
    )

    return latex_str


def _resolve_legend_anchor_y(
    axes_top: float,
    legend_anchor_y: Optional[float],
    legend_gap: float,
) -> float:
    """
    Resolve legend y-position.

    If legend_anchor_y is explicitly given, use it.
    Otherwise place the legend `legend_gap` above the subplot area.
    """
    if legend_anchor_y is not None:
        return float(legend_anchor_y)
    return float(axes_top + legend_gap)


def plot_dataset_comparisons(
    dataset_folders: List[str],
    algorithms: List[str] = ["TPE", "HB", "ES", "PLATEAU"],
    tune_criterion: bool = False,
    depth_trees_only: bool = False,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    deltas: Union[float, List[float]] = DEFAULT_DELTA_GRID[0],
    n_trials: int = DEFAULT_N_TRIALS_GRID[-1],
    *,
    ncols: Optional[int] = None,
    bw: float = 0.82,
    group_gap: float = 1.2,
    min_w: float = 3.2,
    min_h: float = 2.6,
    titles: Optional[List[str]] = None,
    save_plots: bool = True,
    # --- layout ---
    layout_left: float = 0.01,
    layout_right: float = 0.99,
    layout_bottom: float = 0.01,
    axes_top: float = 0.9,
    wspace: float = 0.4,
    hspace: float = 0.2,
    # --- legend ---
    legend_anchor_y: Optional[float] = None,
    legend_gap: float = 0.1,
    alg_legend_x: float = 0.36,
    metric_legend_x: float = 0.78,
) -> None:
    """
    Plot grouped comparisons of tuning time and selected tree count across datasets.

    For each dataset, this function creates one subplot with two bar groups:
    total tuning time on the left y-axis and the selected number of trees ``T``
    on the right y-axis. Within each metric group, bars correspond to the
    requested algorithms, shown in the canonical order ``TPE``, ``HB``, ``ES``,
    ``PLATEAU`` whenever these algorithms are included.

    The function reads experiment summaries from directories resolved by
    ``get_experiment_directory()`` and aggregated by ``read_experiment_results()``.
    A subplot is hidden if the corresponding experiment directory is missing or
    if the required metrics cannot be extracted.

    Args:
        dataset_folders: Dataset-level experiment folders. Each entry is passed
            to ``get_experiment_directory()`` together with the selected
            algorithm and configuration flags.
        algorithms: Algorithms to show. Supported values are ``"TPE"``,
            ``"HB"``, ``"ES"``, and ``"PLATEAU"``. The displayed order is always
            canonical, independent of the input order.
        tune_criterion: Whether to use experiment folders where the Random
            Forest split criterion was tuned.
        depth_trees_only: Whether to use experiment folders corresponding to
            the restricted search space with tree depth and tree count only.
        scale_factor: Plateau scale factor used to resolve PLATEAU experiment
            directories.
        deltas: Plateau tolerance value(s). If a scalar is given, the same
            tolerance is used for all datasets. If a list/tuple/array is given,
            ``deltas[i]`` is used for ``dataset_folders[i]``.
        n_trials: Number of HPO trials used to resolve experiment directories.

        ncols: Number of subplot columns. If ``None``, a nearly square layout is
            chosen automatically as ``ceil(sqrt(n_datasets))``.
        bw: Width of each individual bar in data coordinates. Larger values
            make bars thicker and reduce the visible gap between neighboring
            algorithms inside the same metric group. This parameter does not
            change the distance between the time group and the tree-count group;
            use ``group_gap`` for that.
        group_gap: Horizontal gap, in data coordinates, between the time bars
            and the tree-count bars inside each subplot. Increasing this value
            separates the two metric groups more clearly; decreasing it makes
            the plot more compact.
        min_w: Minimum width, in inches, allocated to each subplot column. The
            total figure width is computed as ``ncols * min_w``. Increase this
            value if algorithm groups or y-axis labels look compressed.
        min_h: Minimum height, in inches, allocated to each subplot row. The
            total figure height is computed as ``nrows * min_h + 0.9``. Increase
            this value if subplot titles, tick labels, or legends overlap.
        titles: Optional custom titles for subplots. If provided, its length
            must match ``dataset_folders``. If omitted, folder names are used.
        save_plots: Passed to ``read_experiment_results()``. If ``True``, the
            reader may save its auxiliary diagnostic plots while loading each
            experiment folder.

        layout_left: Left boundary of the subplot area in normalized figure
            coordinates, passed to ``fig.subplots_adjust(left=...)``. Values are
            in the interval ``[0, 1]``; increase this value to reserve more space
            for left y-axis labels.
        layout_right: Right boundary of the subplot area in normalized figure
            coordinates, passed to ``fig.subplots_adjust(right=...)``. Decrease
            this value to reserve more space for right y-axis labels or external
            objects.
        layout_bottom: Bottom boundary of the subplot area in normalized figure
            coordinates. Increase this value when x-axis labels or annotations
            at the bottom are clipped.
        axes_top: Top boundary of the subplot area in normalized figure
            coordinates. This is also used as the reference level for automatic
            legend placement. Lower values create more space above the axes.
        wspace: Horizontal spacing between subplot columns, passed to
            ``fig.subplots_adjust(wspace=...)``.
        hspace: Vertical spacing between subplot rows, passed to
            ``fig.subplots_adjust(hspace=...)``.

        legend_anchor_y: Absolute y-coordinate of the legend anchor in
            normalized figure coordinates. If ``None``, the legend is placed at
            ``axes_top + legend_gap``.
        legend_gap: Vertical offset between ``axes_top`` and the legend anchor
            when ``legend_anchor_y`` is not provided. Increase this value if the
            legend overlaps subplot titles.
        alg_legend_x: Horizontal anchor position of the algorithm legend in
            normalized figure coordinates.
        metric_legend_x: Horizontal anchor position of the metric legend in
            normalized figure coordinates.

    Returns:
        None. The function creates a Matplotlib figure and displays it with
        ``plt.show()``.

    Raises:
        ValueError: If an unknown algorithm is requested or if custom ``titles``
            do not match the number of datasets.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    # ---------------- helpers ----------------
    def _delta_for_idx(i: int) -> float:
        if isinstance(deltas, (list, tuple, np.ndarray)):
            return float(deltas[i])
        return float(deltas)

    def _has_metrics(res: Dict[str, Any], keys: List[str]) -> bool:
        return all((k in res) and (res[k] is not None) for k in keys)

    def _clipped_yerr(means, stds):
        """
        Make yerr asymmetric so that bars never imply negatives.
        """
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        lower = np.minimum(stds, means)
        upper = stds
        return np.vstack([lower, upper])

    # ---------------- validation ----------------
    allowed_algs = {"TPE", "HB", "ES", "PLATEAU"}
    unknown = [a for a in algorithms if a not in allowed_algs]
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")

    if titles is not None and len(titles) != len(dataset_folders):
        raise ValueError(
            f"Length of titles ({len(titles)}) must match length of dataset_folders ({len(dataset_folders)})"
        )

    canonical_order = ["TPE", "HB", "ES", "PLATEAU"]
    algorithms = [a for a in canonical_order if a in algorithms]

    required_metrics = [
        "time_total",
        "time_total_mean",
        "time_total_std",
        "BEST_n_estimators",
        "BEST_n_estimators_mean",
        "BEST_n_estimators_std",
    ]

    # ---------------- common style ----------------
    TITLE_FONT_SIZE = 12
    AXIS_LABEL_FONT_SIZE = 12
    TICK_LABEL_FONT_SIZE = 11
    LEGEND_FONT_SIZE = 11

    alg_colors = {
        "TPE": "#4C72B0",      # blue
        "HB": "#DD8452",       # orange
        "ES": "#8C8C8C",       # gray
        "PLATEAU": "#55A868",  # green
    }

    hatch_time = ""
    hatch_t = "///"

    # ---------------- layout ----------------
    n_datasets = len(dataset_folders)
    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)

    fig_width = ncols * min_w
    fig_height = nrows * min_h + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_arr = np.atleast_2d(axes) if isinstance(axes, np.ndarray) else np.array([[axes]])
    if axes_arr.shape != (nrows, ncols):
        axes_arr = axes_arr.reshape(nrows, ncols)
    axes_list = axes_arr.ravel().tolist()

    # ---------------- legend handles ----------------
    alg_legend_handles = [
        Patch(facecolor=alg_colors[a], edgecolor="black", label=a) for a in algorithms
    ]
    metric_legend_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_time, label="Time"),
        Patch(facecolor="white", edgecolor="black", hatch=hatch_t, label="$T$"),
    ]

    # ---------------- plot per dataset ----------------
    for idx, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[idx]
        delta_i = _delta_for_idx(idx)

        results: Dict[str, Dict[str, Any]] = {}
        ok = True
        for alg in algorithms:
            try:
                folder = get_experiment_directory(
                    dataset_folder,
                    tune_criterion,
                    depth_trees_only,
                    alg,
                    scale_factor=scale_factor,
                    delta=delta_i,
                    n_trials=n_trials,
                )
                results[alg] = read_experiment_results(folder, save_plots=save_plots)
            except (FileNotFoundError, KeyError):
                ok = False
                break

        if not ok or not all(_has_metrics(results[alg], required_metrics) for alg in algorithms):
            ax.set_visible(False)
            continue

        time_means = [results[alg]["time_total_mean"] for alg in algorithms]
        time_stds = [results[alg]["time_total_std"] for alg in algorithms]

        t_means = [results[alg]["BEST_n_estimators_mean"] for alg in algorithms]
        t_stds = [results[alg]["BEST_n_estimators_std"] for alg in algorithms]

        n_algs = len(algorithms)

        x_time = np.arange(n_algs, dtype=float)
        x_t = x_time + n_algs + group_gap

        # ---- Time bars (left axis)
        for i_alg, alg in enumerate(algorithms):
            ax.bar(
                x_time[i_alg],
                time_means[i_alg],
                yerr=_clipped_yerr([time_means[i_alg]], [time_stds[i_alg]]),
                width=bw,
                color=alg_colors[alg],
                hatch=hatch_time,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )

        ax.set_ylim(bottom=0)
        ax.set_ylabel("Time (s)", fontsize=AXIS_LABEL_FONT_SIZE)

        # ---- T bars (right axis)
        ax2 = ax.twinx()
        for i_alg, alg in enumerate(algorithms):
            ax2.bar(
                x_t[i_alg],
                t_means[i_alg],
                yerr=_clipped_yerr([t_means[i_alg]], [t_stds[i_alg]]),
                width=bw,
                color=alg_colors[alg],
                hatch=hatch_t,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )

        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("$T$", fontsize=AXIS_LABEL_FONT_SIZE)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xticks([])
        ax.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONT_SIZE)
        ax2.tick_params(axis="y", which="major", labelsize=TICK_LABEL_FONT_SIZE)
        ax.tick_params(axis="x", which="major", labelsize=TICK_LABEL_FONT_SIZE)

        title = Path(dataset_folder).name if titles is None else titles[idx]
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax2.grid(False)

        ax.set_xlim(-0.8, x_t[-1] + 0.8)

    # ---------------- hide unused axes ----------------
    for j in range(len(dataset_folders), len(axes_list)):
        axes_list[j].set_visible(False)

    legend_y = _resolve_legend_anchor_y(
        axes_top=axes_top,
        legend_anchor_y=legend_anchor_y,
        legend_gap=legend_gap,
    )

    # ---------------- legends ----------------
    leg1 = fig.legend(
        handles=alg_legend_handles,
        loc="upper center",
        ncol=len(algorithms),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(alg_legend_x, legend_y),
        title="Algorithm",
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.add_artist(leg1)

    fig.legend(
        handles=metric_legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(metric_legend_x, legend_y),
        title="Metric",
        title_fontsize=LEGEND_FONT_SIZE,
    )

    # ---------------- unified layout ----------------
    fig.subplots_adjust(
        left=layout_left,
        right=layout_right,
        bottom=layout_bottom,
        top=axes_top,
        wspace=wspace,
        hspace=hspace,
    )

    plt.show()


def plot_delta_boxplots(
    dataset_folders: List[str],
    deltas_grid,
    *,
    tune_criterion: bool = False,
    depth_trees_only: bool = False,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    n_trials: int = DEFAULT_N_TRIALS_GRID[-1],
    ncols: Optional[int] = None,
    min_w: float = 3.2,
    min_h: float = 2.6,
    box_width: float = 0.6,
    show_means: bool = True,
    mean_marker: str = "o",
    mean_marker_size: int = 18,
    titles: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    save_plots_from_reader: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    # --- layout ---
    layout_left: float = 0.01,
    layout_right: float = 0.99,
    layout_bottom: float = 0.01,
    axes_top: float = 0.90,
    wspace: float = 0.32,
    hspace: float = 0.40,
) -> None:
    """
    Plot boxplots of best scores as a function of the plateau tolerance.

    For each dataset, this function creates one subplot with boxplots of
    ``BEST_score`` grouped by the PLATEAU tolerance ``epsilon`` (called
    ``delta`` in the experiment directory structure). Each box summarizes the
    distribution of best scores over repeated runs for a fixed tolerance value.
    Optionally, the empirical mean score for each tolerance is overlaid as a
    marker.

    The x-axis is formatted for the common tolerance grid
    ``1e-3, 3e-3, 5e-3, ...`` by showing integer multipliers on the ticks and
    placing a shared ``x 10^{-3}`` multiplier annotation near the lower-right
    side of each subplot.

    Args:
        dataset_folders: Dataset-level experiment folders. For each dataset and
            each tolerance value, the PLATEAU experiment directory is resolved
            using ``get_experiment_directory()``.
        deltas_grid: Plateau tolerance values. This can be either a single list
            shared by all datasets, e.g. ``[1e-3, 3e-3, 5e-3]``, or a list of
            lists where ``deltas_grid[i]`` contains the tolerance values for
            ``dataset_folders[i]``.
        tune_criterion: Whether to use experiment folders where the Random
            Forest split criterion was tuned.
        depth_trees_only: Whether to use experiment folders corresponding to
            the restricted search space with tree depth and tree count only.
        scale_factor: Plateau scale factor used to resolve experiment
            directories.
        n_trials: Number of HPO trials used to resolve experiment directories.

        ncols: Number of subplot columns. If ``None``, a nearly square layout is
            chosen automatically as ``ceil(sqrt(n_datasets))``.
        min_w: Minimum width, in inches, allocated to each subplot column. The
            total figure width is computed as ``ncols * min_w``.
        min_h: Minimum height, in inches, allocated to each subplot row. The
            total figure height is computed as ``nrows * min_h + 0.8``.
        box_width: Width of each boxplot in data coordinates. Smaller values
            create more horizontal whitespace around boxes; larger values make
            boxes visually heavier.
        show_means: Whether to overlay the empirical mean score for each
            tolerance value.
        mean_marker: Matplotlib marker style used for the overlaid means.
        mean_marker_size: Marker size used for the overlaid means.
        titles: Optional custom subplot titles. If provided, its length must
            match ``dataset_folders``. If omitted, folder names are used.
        ylabels: Optional custom y-axis labels, one per dataset. If omitted,
            ``"Best score"`` is used for all subplots.
        save_plots_from_reader: Passed to ``read_experiment_results()``. If
            ``True``, the reader may save its auxiliary diagnostic plots while
            loading each experiment folder.
        save_path: Optional path where the final figure is saved. Parent
            directories are created automatically.
        dpi: Resolution used when saving the figure.

        layout_left: Left boundary of the subplot area in normalized figure
            coordinates, passed to ``fig.subplots_adjust(left=...)``. Increase
            this value to reserve more space for y-axis labels.
        layout_right: Right boundary of the subplot area in normalized figure
            coordinates, passed to ``fig.subplots_adjust(right=...)``.
        layout_bottom: Bottom boundary of the subplot area in normalized figure
            coordinates. Increase this value if the x-axis label or the
            ``x 10^{-3}`` multiplier annotation is clipped.
        axes_top: Top boundary of the subplot area in normalized figure
            coordinates. Lower values reserve more space above the subplots.
        wspace: Horizontal spacing between subplot columns, passed to
            ``fig.subplots_adjust(wspace=...)``.
        hspace: Vertical spacing between subplot rows, passed to
            ``fig.subplots_adjust(hspace=...)``.

    Returns:
        None. The function creates a Matplotlib figure, optionally saves it, and
        displays it with ``plt.show()``.

    Raises:
        ValueError: If ``deltas_grid`` is empty, if per-dataset tolerance lists
            do not match the number of datasets, or if custom ``titles`` or
            ``ylabels`` have inconsistent lengths.
    """
    if titles is not None and len(titles) != len(dataset_folders):
        raise ValueError("If provided, titles must have the same length as dataset_folders.")
    if ylabels is not None and len(ylabels) != len(dataset_folders):
        raise ValueError("If provided, ylabels must have the same length as dataset_folders.")

    n_datasets = len(dataset_folders)

    if len(deltas_grid) == 0:
        raise ValueError("deltas_grid must be non-empty.")

    if isinstance(deltas_grid[0], (list, tuple, np.ndarray)):
        if len(deltas_grid) != n_datasets:
            raise ValueError(
                f"deltas_grid has length {len(deltas_grid)}, but dataset_folders has length {n_datasets}. "
                "Provide one delta-list per dataset."
            )
        deltas_per_dataset = [list(map(float, ds_deltas)) for ds_deltas in deltas_grid]
    else:
        shared = list(map(float, deltas_grid))
        deltas_per_dataset = [shared for _ in range(n_datasets)]

    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)

    fig_width = ncols * min_w
    fig_height = nrows * min_h + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_arr = np.atleast_2d(axes) if isinstance(axes, np.ndarray) else np.array([[axes]])
    if axes_arr.shape != (nrows, ncols):
        axes_arr = axes_arr.reshape(nrows, ncols)
    axes_list = axes_arr.ravel().tolist()

    TITLE_FONT_SIZE = 12
    AXIS_LABEL_FONT_SIZE = 12
    TICK_LABEL_FONT_SIZE = 11

    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[i]
        ds_deltas = deltas_per_dataset[i]

        scores_by_delta: List[List[float]] = []
        means: List[float] = []

        for d in ds_deltas:
            path = get_experiment_directory(
                dataset_folder,
                tune_criterion,
                depth_trees_only,
                method="PLATEAU",
                scale_factor=scale_factor,
                delta=float(d),
                n_trials=n_trials,
            )

            try:
                res = read_experiment_results(path, save_plots=save_plots_from_reader)
            except (FileNotFoundError, KeyError):
                continue

            vals = res.get("BEST_score")
            if vals is None:
                continue

            vals = [x for x in vals if x is not None and not np.isnan(x)]
            if len(vals) == 0:
                continue

            scores_by_delta.append(vals)
            means.append(float(np.mean(vals)))

        if len(scores_by_delta) == 0:
            ax.set_title(Path(dataset_folder).name if titles is None else titles[i], fontsize=TITLE_FONT_SIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if ylabels is not None:
                ax.set_ylabel(ylabels[i], fontsize=AXIS_LABEL_FONT_SIZE)
            continue

        positions = np.arange(1, len(scores_by_delta) + 1, dtype=float)

        bp = ax.boxplot(
            scores_by_delta,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(linewidth=1.6),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )

        for b in bp["boxes"]:
            b.set_facecolor("white")

        if show_means:
            ax.scatter(
                positions,
                means,
                marker=mean_marker,
                s=mean_marker_size,
                zorder=3,
            )

        scale = 1e-3
        multipliers = [int(round(d / scale)) for d in ds_deltas[:len(scores_by_delta)]]
        ax.set_xticks(positions)
        ax.set_xticklabels([str(m) for m in multipliers], fontsize=TICK_LABEL_FONT_SIZE)

        ax.text(
            0.95, -0.18, r'$\times 10^{-3}$',
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=TICK_LABEL_FONT_SIZE
        )

        ax.set_xlabel(r"$\varepsilon$", fontsize=AXIS_LABEL_FONT_SIZE)

        if ylabels is not None:
            ax.set_ylabel(ylabels[i], fontsize=AXIS_LABEL_FONT_SIZE)
        else:
            ax.set_ylabel("Best score", fontsize=AXIS_LABEL_FONT_SIZE)

        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)

        ax.set_title(Path(dataset_folder).name if titles is None else titles[i], fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)

    for j in range(len(dataset_folders), len(axes_list)):
        axes_list[j].set_visible(False)

    fig.subplots_adjust(
        left=layout_left,
        right=layout_right,
        bottom=layout_bottom,
        top=axes_top,
        wspace=wspace,
        hspace=hspace,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    plt.show()


def _select_even_ticks_from_levels(levels, max_ticks: int = 10):
    """
    Select at most `max_ticks` labels from a sorted array of true B-levels,
    using a constant integer step in level-index coordinates.

    The highest level is always included. The number of shown ticks may be
    smaller than `max_ticks`.
    """
    levels = np.asarray(levels)
    n = len(levels)

    if n == 0:
        return np.array([], dtype=int), []

    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive.")

    step = max(1, int(np.ceil(n / max_ticks)))
    offset = (n - 1) % step
    idx = np.arange(offset, n, step, dtype=int)

    return idx, [str(int(levels[i])) for i in idx]


def _choose_xticks(n_trials_total: int, max_ticks: int) -> np.ndarray:
    """
    Choose trial ticks in 1..n_trials_total with a constant integer step.

    The last tick n_trials_total is always included. The number of shown ticks
    may be smaller than `max_ticks`.
    """
    if n_trials_total <= 0:
        return np.array([], dtype=int)

    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive.")

    step = max(1, int(np.ceil(n_trials_total / max_ticks)))
    offset = (n_trials_total - 1) % step
    return np.arange(offset + 1, n_trials_total + 1, step, dtype=int)


def _build_nominal_B_levels(
    scale_factor: float,
    n_estimators_start: int,
    max_trees: int,
    min_observed: int,
    max_observed: int,
) -> np.ndarray:
    """
    Build the nominal B-grid around n_estimators_start.

    Upward levels are built using build_ladder(scale_factor, n_estimators_start, max_trees),
    then truncated so that the highest retained level is at most one ladder step above
    the observed maximum.

    Downward levels are generated by repeated division by scale_factor (with rounding)
    until the minimum observed value is covered.
    """
    if min_observed <= 0 or max_observed <= 0:
        raise ValueError("Observed B values must be positive.")

    up = np.asarray(build_ladder(scale_factor, n_estimators_start, max_trees), dtype=int)

    cutoff = int(np.searchsorted(up, max_observed, side="right"))
    cutoff = min(cutoff, len(up) - 1)
    up = up[: cutoff + 1]

    down = [int(n_estimators_start)]
    while down[-1] > min_observed:
        nxt = int(round(down[-1] / scale_factor))
        if nxt >= down[-1] or nxt < 1:
            break
        down.append(nxt)

    down = np.asarray(sorted(set(down)), dtype=int)
    levels = np.asarray(sorted(set(down.tolist() + up.tolist())), dtype=int)
    return levels


def plot_B_trajectories(
    dataset_folders: List[str],
    deltas: Union[float, List[float]] = DEFAULT_DELTA_GRID[0],
    *,
    tune_criterion: bool = False,
    depth_trees_only: bool = False,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    n_trials: int = DEFAULT_N_TRIALS_GRID[-1],
    run_idx: int = 0,
    ncols: Optional[int] = None,
    min_w: float = 3.2,
    min_h: float = 2.6,
    titles: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    save_plots_from_reader: bool = False,
    cmap: str = "Blues",
    show_colorbar: bool = True,
    show_mean_path: bool = True,
    line_color: str = "black",
    mean_color: str = "orange",
    max_yticks: int = 10,
    max_xticks: int = 7,
    # --- layout ---
    layout_left: float = 0.01,
    layout_right: float = 0.99,
    layout_bottom: float = 0.01,
    axes_top: float = 0.84,
    wspace: float = 0.32,
    hspace: float = 0.40,
    # --- legend ---
    legend_anchor_y: Optional[float] = None,
    legend_gap: float = 0.04,
    # --- colorbar geometry ---
    cbar_gap: float = 0.010,
    cbar_width: float = 0.012,
) -> None:
    """
    Plot trial-wise trajectories of the central triplet point ``B``.

    For each dataset, this function visualizes the evolution of the central
    tree count ``B`` in the PLATEAU algorithm. The selected run is shown as a
    line trajectory, and the background heatmap shows the empirical probability
    of observing each nominal ``B`` level at each trial across all available
    repeated runs.

    The vertical axis is not a continuous numeric tree-count axis. Instead, it
    is an index over the nominal geometric grid generated from
    ``n_estimators_start``, ``scale_factor``, and ``max_trees``. Tick labels are
    then replaced by the corresponding actual tree-count values. This keeps the
    cells square and makes geometric movements of the triplet visually clear.

    Args:
        dataset_folders: Dataset-level experiment folders. For each dataset,
            the corresponding PLATEAU experiment directory is resolved using
            ``get_experiment_directory()``.
        deltas: Plateau tolerance value(s). If a scalar is given, the same
            tolerance is used for all datasets. If a list/tuple/array is given,
            ``deltas[i]`` is used for ``dataset_folders[i]``.
        tune_criterion: Whether to use experiment folders where the Random
            Forest split criterion was tuned.
        depth_trees_only: Whether to use experiment folders corresponding to
            the restricted search space with tree depth and tree count only.
        scale_factor: Plateau scale factor used to resolve experiment
            directories.
        n_trials: Number of HPO trials used to resolve experiment directories.
        run_idx: Index of the repeated run whose individual trajectory is drawn
            on top of the probability map.

        ncols: Number of subplot columns. If ``None``, a nearly square layout is
            chosen automatically as ``ceil(sqrt(n_datasets))``.
        min_w: Minimum width, in inches, allocated to each subplot column. The
            total figure width is computed as ``ncols * min_w``.
        min_h: Minimum height, in inches, allocated to each subplot row. The
            total figure height is computed as ``nrows * min_h + 0.9``.
        titles: Optional custom subplot titles. If provided, its length must
            match ``dataset_folders``. If omitted, folder names are used.
        save_path: Optional path where the final figure is saved. Parent
            directories are created automatically.
        dpi: Resolution used when saving the figure.
        save_plots_from_reader: Passed to ``read_experiment_results()``. If
            ``True``, the reader may save its auxiliary diagnostic plots while
            loading each experiment folder.
        cmap: Matplotlib colormap name used for the background probability map.
        show_colorbar: Whether to draw one shared probability colorbar per
            subplot row.
        show_mean_path: Whether to overlay the across-run mean trajectory in
            nominal grid-index coordinates.
        line_color: Color of the selected individual run trajectory.
        mean_color: Color of the across-run mean trajectory.
        max_yticks: Maximum number of nominal ``B`` levels shown as y-axis tick
            labels. The actual number may be smaller because ticks are selected
            on the discrete level grid.
        max_xticks: Maximum number of trial ticks shown on the x-axis. The final
            trial is always included when possible.

        layout_left: Left boundary of the subplot area in normalized figure
            coordinates, passed to ``fig.subplots_adjust(left=...)``. Increase
            this value to reserve more space for y-axis labels.
        layout_right: Right boundary of the subplot area in normalized figure
            coordinates. Decrease this value to reserve space for row-level
            colorbars on the right.
        layout_bottom: Bottom boundary of the subplot area in normalized figure
            coordinates. Increase this value if x-axis labels are clipped.
        axes_top: Top boundary of the subplot area in normalized figure
            coordinates. This is also used as the reference level for automatic
            legend placement.
        wspace: Horizontal spacing between subplot columns, passed to
            ``fig.subplots_adjust(wspace=...)``.
        hspace: Vertical spacing between subplot rows, passed to
            ``fig.subplots_adjust(hspace=...)``.

        legend_anchor_y: Absolute y-coordinate of the legend anchor in
            normalized figure coordinates. If ``None``, the legend is placed at
            ``axes_top + legend_gap``.
        legend_gap: Vertical offset between ``axes_top`` and the legend anchor
            when ``legend_anchor_y`` is not provided. Increase this value if the
            legend overlaps subplot titles.

        cbar_gap: Horizontal gap, in normalized figure coordinates, between the
            right edge of each visible subplot row and the corresponding
            colorbar.
        cbar_width: Width of each row-level colorbar in normalized figure
            coordinates.

    Returns:
        None. The function creates a Matplotlib figure, optionally saves it, and
        displays it with ``plt.show()``.

    Raises:
        ValueError: If custom ``titles`` do not match the number of datasets, if
            ``run_idx`` is outside the available range for a dataset, or if the
            parsed experiment results do not contain the metadata required to
            reconstruct the nominal ``B`` grid.
    """
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    def _delta_for_idx(i: int) -> float:
        if isinstance(deltas, (list, tuple, np.ndarray)):
            return float(deltas[i])
        return float(deltas)

    def _nearest_level_idx(val: float, levels: np.ndarray) -> int:
        return int(np.argmin(np.abs(levels - float(val))))

    if titles is not None and len(titles) != len(dataset_folders):
        raise ValueError(
            f"Length of titles ({len(titles)}) must match length of dataset_folders ({len(dataset_folders)})"
        )

    n_datasets = len(dataset_folders)
    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)

    fig_width = ncols * min_w
    fig_height = nrows * min_h + 0.9
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_arr = np.atleast_2d(axes) if isinstance(axes, np.ndarray) else np.array([[axes]])
    if axes_arr.shape != (nrows, ncols):
        axes_arr = axes_arr.reshape(nrows, ncols)
    axes_list = axes_arr.ravel().tolist()

    TITLE_FONT_SIZE = 12
    AXIS_LABEL_FONT_SIZE = 12
    TICK_LABEL_FONT_SIZE = 11
    LEGEND_FONT_SIZE = 11

    norm = Normalize(vmin=0.0, vmax=1.0)
    row_visible_axes = [[] for _ in range(nrows)]

    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[i]
        row_idx = i // ncols
        delta_i = _delta_for_idx(i)

        path = get_experiment_directory(
            dataset_folder,
            tune_criterion,
            depth_trees_only,
            method="PLATEAU",
            scale_factor=scale_factor,
            delta=delta_i,
            n_trials=n_trials,
        )

        try:
            res = read_experiment_results(path, save_plots=save_plots_from_reader)
        except (FileNotFoundError, KeyError):
            ax.set_visible(False)
            continue

        B_all = res.get("B")
        n_trials_total = res.get("n_trials")
        n_estimators_start_ = res.get("n_estimators_start")
        max_trees_ = res.get("max_trees")
        scale_factor_ = res.get("scale_factor")

        if B_all is None or len(B_all) == 0:
            ax.set_title(Path(dataset_folder).name if titles is None else titles[i], fontsize=TITLE_FONT_SIZE)
            ax.text(0.5, 0.5, "No B data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if not (0 <= run_idx < len(B_all)):
            raise ValueError(
                f"run_idx={run_idx} is out of range for dataset '{dataset_folder}'. "
                f"Available runs: 0..{len(B_all)-1}"
            )

        B_rows = []
        max_len = 0
        positive_vals = []

        for row in B_all:
            if row is None:
                continue
            arr = np.asarray(row, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            B_rows.append(arr)
            max_len = max(max_len, len(arr))
            positive_vals.extend([int(v) for v in arr if v > 0])

        if len(B_rows) == 0 or max_len == 0 or len(positive_vals) == 0:
            ax.set_title(Path(dataset_folder).name if titles is None else titles[i], fontsize=TITLE_FONT_SIZE)
            ax.text(0.5, 0.5, "Empty B trajectories", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if n_trials_total is None:
            n_trials_total = max_len
        else:
            n_trials_total = int(n_trials_total)

        if n_estimators_start_ is None or max_trees_ is None or scale_factor_ is None:
            raise ValueError("Missing n_estimators_start / max_trees / scale_factor in parsed results.")

        min_observed = int(min(positive_vals))
        max_observed = int(max(positive_vals))

        levels = _build_nominal_B_levels(
            scale_factor=float(scale_factor_),
            n_estimators_start=int(n_estimators_start_),
            max_trees=int(max_trees_),
            min_observed=min_observed,
            max_observed=max_observed,
        )
        n_levels = len(levels)

        prob = np.zeros((n_levels, n_trials_total), dtype=float)
        counts = np.zeros(n_trials_total, dtype=float)

        # --- trajectories converted to level indices ---
        idx_mat = np.full((len(B_rows), n_trials_total), np.nan, dtype=float)

        for r, row in enumerate(B_rows):
            L = min(len(row), n_trials_total)
            for t in range(L):
                val = row[t]
                if np.isfinite(val) and val > 0:
                    idx = _nearest_level_idx(val, levels)
                    idx_mat[r, t] = idx
                    prob[idx, t] += 1.0
                    counts[t] += 1.0

        valid = counts > 0
        prob[:, valid] /= counts[valid][None, :]

        # --- selected run path in level-index coordinates ---
        B_sel = np.asarray(B_all[run_idx], dtype=float)
        sel_mask = np.isfinite(B_sel) & (B_sel > 0)
        B_sel = B_sel[sel_mask]
        L_sel = min(len(B_sel), n_trials_total)
        x_sel = np.arange(1, L_sel + 1, dtype=int)
        y_sel = np.array([_nearest_level_idx(v, levels) for v in B_sel[:L_sel]], dtype=float)

        # --- square-cell probability map ---
        x_edges = np.arange(0.5, n_trials_total + 1.5, 1.0)
        y_edges = np.arange(-0.5, n_levels + 0.5, 1.0)
        prob_masked = np.ma.masked_where(prob <= 0, prob)

        ax.pcolormesh(
            x_edges,
            y_edges,
            prob_masked,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )

        # --- selected run on top ---
        ax.plot(
            x_sel,
            y_sel,
            color=line_color,
            linewidth=1.5,
            marker="o",
            markersize=3,
            alpha=0.95,
            label=f"Run #{run_idx}",
        )

        # --- mean path in level-index coordinates ---
        if show_mean_path:
            mean_idx = np.nanmean(idx_mat, axis=0)
            mean_mask = np.isfinite(mean_idx)
            if np.any(mean_mask):
                x_mean = np.arange(1, n_trials_total + 1, dtype=int)[mean_mask]
                ax.plot(
                    x_mean,
                    mean_idx[mean_mask],
                    color=mean_color,
                    linewidth=2.5,
                    linestyle="-",
                    alpha=0.95,
                    label="Mean path",
                )

        ax.set_xlabel("Trial", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("$B$", fontsize=AXIS_LABEL_FONT_SIZE)

        y_tick_pos, y_tick_labels = _select_even_ticks_from_levels(levels, max_ticks=max_yticks)
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(y_tick_labels, fontsize=TICK_LABEL_FONT_SIZE)

        x_ticks = _choose_xticks(n_trials_total, max_ticks=max_xticks)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks], fontsize=TICK_LABEL_FONT_SIZE)

        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)

        title = Path(dataset_folder).name if titles is None else titles[i]
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(False)

        row_visible_axes[row_idx].append(ax)

    for j in range(len(dataset_folders), len(axes_list)):
        axes_list[j].set_visible(False)

    # Finalize subplot geometry
    fig.subplots_adjust(
        left=layout_left,
        right=layout_right,
        bottom=layout_bottom,
        top=axes_top,
        wspace=wspace,
        hspace=hspace,
    )

    # Add one colorbar per row after subplot positions are final
    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        for r in range(nrows):
            row_axes = row_visible_axes[r]
            if not row_axes:
                continue

            right = max(ax.get_position().x1 for ax in row_axes)
            y0 = min(ax.get_position().y0 for ax in row_axes)
            y1 = max(ax.get_position().y1 for ax in row_axes)

            cax = fig.add_axes([right + cbar_gap, y0, cbar_width, y1 - y0])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=TICK_LABEL_FONT_SIZE)
            cbar.set_label("Probability", fontsize=AXIS_LABEL_FONT_SIZE)

    legend_y = _resolve_legend_anchor_y(
        axes_top=axes_top,
        legend_anchor_y=legend_anchor_y,
        legend_gap=legend_gap,
    )

    legend_handles = [
        Line2D([0], [0], color=line_color, marker="o", markersize=4, linewidth=1.5, label=f"Random trajectory")
    ]
    if show_mean_path:
        legend_handles.append(
            Line2D([0], [0], color=mean_color, linewidth=2.0, linestyle="-", label="Mean trajectory")
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    plt.show()