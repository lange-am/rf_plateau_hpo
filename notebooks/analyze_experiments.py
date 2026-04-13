"""
analyze_experiments.py
----------------------

This module provides functions for analyzing and visualizing results from Random Forest tuning experiments
performed with `rf_plateau_hpo.core` and orchestrated by `run_experiments.py`.

Key Functions
-------------
- `read_experiment_results()`           – Load and aggregate metrics from `.dill` experiment files.
- `bootstrap_effect_size_alternative()` – Compute Cliff's Delta or Cohen's d with bootstrap confidence intervals.
- `experiment_comparison_table()`       – Generate a comparison table with statistical tests (t-test/Mann‑Whitney)
                                          and effect sizes between different experimental configurations.
- `process_html_table()`                – Produce ultra‑compact HTML table formatting for comparison results.
- `tab2tex()`                           – Convert a comparison DataFrame to a LaTeX `tabular` environment.
- `plot_dataset_comparisons()`          – Create grouped bar plots comparing time and tree counts across datasets.
- `plot_delta_boxplots()`               – Boxplots of best scores versus the plateau tolerance ε (delta).

Additional Utilities
--------------------
- `bootstrap_effect_size_alternative()` – Robust effect size estimation with bootstrap.

All visualizations use Matplotlib and Seaborn. For details on the underlying tuning algorithms,
see the `rf_plateau_hpo.core` module and the companion `run_experiments.py`.

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

from run_experiments import (
    RF_HPO_ALGORITHMS,
    DEFAULT_N_TRIALS_GRID,
    DEFAULT_DELTA_GRID,
    DEFAULT_SCALE_FACTOR_GRID,
    get_experiment_directory,
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

            for k in res.keys():
                if k in ['problem', 'greater_is_better']:
                    res[k] += [params_in.get(k)]
                elif k =='n':
                    res[k] += [n]
                elif k =='p':
                    res[k] += [p]
                elif k =='dataset':
                    res[k] += [dataset]
                else:
                    value = params_out.get(k) or params_out.get('study', {}).get(k)
                    res[k] += [0 if 'n_' in k and value is None else value]

            res['BEST_n_estimators'][-1] = res['BEST_n_estimators'][-1] or res['BEST_params'][-1].get('n_estimators')
        except (KeyError, IOError, dill.UnpicklingError) as e:
            print(f"Error processing {dill_file}: {e}")
            continue

    COMMOM_PARAMS = ['problem', 'greater_is_better', 'n', 'p', 'dataset']
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
        if k not in ['BEST_params'] + COMMOM_PARAMS:
            try:
                if k == 'BEST_params_max_features':
                    mf_sqrt = 1/np.sqrt(res['p']) if res['p'] else None
                    v_ = [mf_sqrt if x == 'sqrt' else x for x in v]
                else:
                    v_ = v
                if any(x is not None for x in v_):
                    res_[f'{k}_mean'] = np.nanmean(np.array(v_, dtype=float))
                    res_[f'{k}_std'] = np.nanstd(np.array(v_, dtype=float))
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
    SIGNIFICANCE = "t-test p-value, Cohen's $d$ (or Mann-Whitney p-value, Cliff's $\delta$)"
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


def plot_dataset_comparisons(
    dataset_folders: List[str],
    algorithms: List[str] = ["TPE", "HB", "PLATEAU"],
    tune_criterion: bool = False,
    depth_trees_only: bool = False,
    scale_factor: float = DEFAULT_SCALE_FACTOR_GRID[0],
    deltas: Union[float, List[float]] = DEFAULT_DELTA_GRID[0],
    n_trials: int = DEFAULT_N_TRIALS_GRID[-1],
    *,
    ncols: Optional[int] = None,
    bw: float = 0.9,
    min_w: float = 3.0,
    min_h: float = 2.5,
    tight_layout_top: float = 0.9,
    save_plots: bool = True,
    titles: Optional[List[str]] = None,
) -> None:
    """
    Compare TPE vs HB vs PLATEAU across multiple datasets.

    For each dataset, the plot shows:
      - Total tuning time (left y-axis, seconds)
      - Selected number of trees T (right y-axis, integer ticks)

    Order (left-to-right within each metric group): TPE, HB, PLATEAU.
    Hatching: TPE='//', HB='\\\\', PLATEAU='' (no hatch).
    """

    # ---------------- helpers ----------------
    def _delta_for_idx(i: int) -> float:
        if isinstance(deltas, (list, tuple, np.ndarray)):
            return float(deltas[i])
        return float(deltas)

    def _has_metrics(res: Dict[str, Any], keys: List[str]) -> bool:
        return all((k in res) and (res[k] is not None) for k in keys)

    def _clipped_yerr(means, stds):
        """
        Matplotlib expands ylim if yerr goes below 0.
        Make yerr asymmetric and clip the lower part so bars never imply negatives.
        """
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        lower = np.minimum(stds, means)  # ensures mean - lower >= 0
        upper = stds
        return np.vstack([lower, upper])  # shape (2, n)

    # ---------------- layout ----------------
    n_datasets = len(dataset_folders)
    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)

    if titles is not None and len(titles) != n_datasets:
        raise ValueError(
            f"Length of titles ({len(titles)}) must match length of dataset_folders ({n_datasets})"
        )

    fig_width = ncols * min_w
    fig_height = nrows * min_h + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_list = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    TITLE_FONT_SIZE = 12
    AXIS_LABEL_FONT_SIZE = 12
    TICK_LABEL_FONT_SIZE = 11
    LEGEND_FONT_SIZE = 12

    COLOR_TIME = "#4C72B0"
    COLOR_T = "#C44E52"

    # visual hatches (requested)
    hatches = {
        "TPE": "//",
        "HB": "\\\\",
        "PLATEAU": "",
    }

    required_metrics = [
        "time_total", "time_total_mean", "time_total_std",
        "BEST_n_estimators", "BEST_n_estimators_mean", "BEST_n_estimators_std",
    ]

    legend_handles, legend_labels = [], []

    # ---------------- plot per dataset ----------------
    for idx, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[idx]
        delta_i = _delta_for_idx(idx)

        # Build result paths and load summaries
        results: Dict[str, Dict[str, Any]] = {}
        try:
            for alg in algorithms:
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
            continue

        if not all(_has_metrics(results[alg], required_metrics) for alg in algorithms):
            continue

        time_means = [results[alg]["time_total_mean"] for alg in algorithms]
        time_stds  = [results[alg]["time_total_std"]  for alg in algorithms]

        t_means = [results[alg]["BEST_n_estimators_mean"] for alg in algorithms]
        t_stds  = [results[alg]["BEST_n_estimators_std"]  for alg in algorithms]

        # x positions: 3 bars for time, gap, 3 bars for T
        x_time = list(range(len(algorithms)))
        x_t    = np.array(x_time) + 3.5

        # ---- Time bars (left axis)
        bars_time = ax.bar(
            x_time,
            time_means,
            yerr=_clipped_yerr(time_means, time_stds),
            width=bw,
            color=COLOR_TIME,
            hatch=[hatches[a] for a in algorithms],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Time (s)", fontsize=AXIS_LABEL_FONT_SIZE)

        # ---- T bars (right axis)
        ax2 = ax.twinx()
        bars_t = ax2.bar(
            x_t,
            t_means,
            yerr=_clipped_yerr(t_means, t_stds),
            width=bw,
            color=COLOR_T,
            hatch=[hatches[a] for a in algorithms],
            edgecolor="black",
            linewidth=0.8,
        )
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("$T$", fontsize=AXIS_LABEL_FONT_SIZE)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # ---- Title
        if titles is None:
            title = Path(dataset_folder).name
        else:
            title = titles[idx]
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

        # ---- cosmetics
        ax.set_xticks([])
        ax.grid(False)
        ax2.grid(False)

        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
        ax2.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)

        # ---- Legend (build once)
        if not legend_handles:
            legend_items = []
            for i, alg in enumerate(algorithms):
                legend_items.append((bars_time[i], f"Time {alg}"))
                legend_items.append((bars_t[i], f"$T$ {alg}"))
            legend_handles, legend_labels = zip(*legend_items)

    # Remove unused axes
    for j in range(len(dataset_folders), len(axes_list)):
        fig.delaxes(axes_list[j])

    # Figure legend
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(algorithms),
            fontsize=LEGEND_FONT_SIZE,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, tight_layout_top])
    plt.show()


def plot_delta_boxplots(
    dataset_folders: List[str],
    deltas_grid,  # Sequence[float] OR Sequence[Sequence[float]]
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
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    titles: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    save_plots_from_reader: bool = False,
) -> None:
    """
    Boxplots of BEST_score versus epsilon (delta), per dataset.

    deltas_grid:
      - either a single list of deltas (applied to all datasets),
      - or a list-of-lists: deltas_grid[i] is used for dataset_folders[i].

    X-axis: delta values (epsilon)
    Y-axis: BEST_score distribution across runs (e.g., 20 seeds)
    Also overlays the mean BEST_score for each delta (optional).

    Parameters
    ----------
    ylabels : Optional[List[str]], default None
        If provided, must have the same length as dataset_folders.
        Each element becomes the y-axis label for the corresponding subplot.
        If None, the default label "Best score" is used for all subplots.
    """

    if titles is not None and len(titles) != len(dataset_folders):
        raise ValueError("If provided, titles must have the same length as dataset_folders.")
    if ylabels is not None and len(ylabels) != len(dataset_folders):
        raise ValueError("If provided, ylabels must have the same length as dataset_folders.")

    n_datasets = len(dataset_folders)

    # Normalize deltas_grid to list-of-lists
    # Accept: deltas_grid = [1e-4, 1e-3, ...]  OR  [[...], [...], ...]
    if len(deltas_grid) == 0:
        raise ValueError("deltas_grid must be non-empty.")

    if isinstance(deltas_grid[0], (list, tuple, np.ndarray)):
        # list-of-lists case
        if len(deltas_grid) != n_datasets:
            raise ValueError(
                f"deltas_grid has length {len(deltas_grid)}, but dataset_folders has length {n_datasets}. "
                "Provide one delta-list per dataset."
            )
        deltas_per_dataset = [list(map(float, ds_deltas)) for ds_deltas in deltas_grid]
    else:
        # single list applied to all datasets
        shared = list(map(float, deltas_grid))
        deltas_per_dataset = [shared for _ in range(n_datasets)]

    # Layout
    ncols = ncols or math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)

    fig_width = ncols * min_w
    fig_height = nrows * min_h + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes_list = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    def _fmt_delta(x: float) -> str:
        # Pretty labels like 1e-3, 3e-4, ...
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / (10 ** exp)
        mant_rounded = np.round(mant, 1)
        if mant_rounded == 1.0:
            return f"$10^{{{exp}}}$"
        return f"${mant_rounded}\\times 10^{{{exp}}}$"

    for i, dataset_folder in enumerate(dataset_folders):
        ax = axes_list[i]
        ds_deltas = deltas_per_dataset[i]

        scores_by_delta: List[List[float]] = []
        means: List[float] = []
        delta_labels: List[str] = []

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
            delta_labels.append(_fmt_delta(float(d)))

        if len(scores_by_delta) == 0:
            ax.set_title(Path(dataset_folder).name if titles is None else titles[i])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            # Set ylabel even for empty subplot if ylabels provided
            if ylabels is not None:
                ax.set_ylabel(ylabels[i])
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

        # Keep boxes unfilled (clean) but visible
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
        multipliers = [int(round(d / scale)) for d in ds_deltas]
        ax.set_xticks(positions)
        ax.set_xticklabels([str(m) for m in multipliers])

        # mapping text: "1→1e-3, 2→3e-3, ..."
        mapping = ", ".join([f"{i}→{lbl}" for i, lbl in enumerate(delta_labels, start=1)])

        # place mapping once per subplot (right side)
        ax.text(
            0.95, -0.25, r'$\times 10^{-3}$',
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=9
        )        
        ax.set_xlabel(r"$\varepsilon$")
        # Set ylabel: custom if provided, else default
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        else:
            ax.set_ylabel("Best score")

        ax.set_title(Path(dataset_folder).name if titles is None else titles[i])
        ax.grid(True, axis="y", alpha=0.25)

    # Remove unused axes
    for j in range(len(dataset_folders), len(axes_list)):
        fig.delaxes(axes_list[j])

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    plt.show()