"""
rf_plateau_hpo
==============

Public API for Random Forest OOB-based hyperparameter optimization with
classic TPE, Hyperband-style, and triplet-based PLATEAU tree-count tuning.
"""

from ._version import __version__
from .core import (
    DEFAULT_DELTA,
    DEFAULT_HYPERBAND_REDUCTION_FACTOR,
    DEFAULT_MAX_DEPTH_RANGE,
    DEFAULT_MAX_FEATURES_GRID,
    DEFAULT_MAX_TREES,
    DEFAULT_MIN_SAMPLES_LEAF_RANGE,
    DEFAULT_MIN_SAMPLES_SPLIT_RANGE,
    DEFAULT_N_ESTIMATORS_LADDER,
    DEFAULT_N_ESTIMATORS_START,
    DEFAULT_N_TRIALS,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    RFCWithOOBProba,
    tune_rf_oob,
    tune_rf_oob_bohb,
    tune_rf_oob_plateau,
)

__all__ = [
    "__version__",
    "RFCWithOOBProba",
    "tune_rf_oob",
    "tune_rf_oob_bohb",
    "tune_rf_oob_plateau",
    "DEFAULT_MAX_FEATURES_GRID",
    "DEFAULT_MAX_DEPTH_RANGE",
    "DEFAULT_MIN_SAMPLES_LEAF_RANGE",
    "DEFAULT_MIN_SAMPLES_SPLIT_RANGE",
    "DEFAULT_N_TRIALS",
    "DEFAULT_T_MIN",
    "DEFAULT_T_MAX",
    "DEFAULT_N_ESTIMATORS_LADDER",
    "DEFAULT_HYPERBAND_REDUCTION_FACTOR",
    "DEFAULT_N_ESTIMATORS_START",
    "DEFAULT_SCALE_FACTOR",
    "DEFAULT_DELTA",
    "DEFAULT_MAX_TREES",
]
