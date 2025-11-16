"""
rf_plateau_hpo.core
-------------------
Two tuners for Random Forest:

1) tune_rf_oob:
   Classic Optuna HPO with OOB objective (scikit-learn OOB support).

2) tune_rf_oob_plateau:
   Triplet-based OOB plateau search that finds a near-minimal `n_estimators`
   while running Bayesian HPO (Optuna) over the other hyperparameters.

Both functions share consistent logging/verbosity and return signatures.
They are compatible with Python 3.8+ (no PEP 585 typing).
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
try:
    from typing import Literal  # Python 3.8+
except Exception:
    from typing_extensions import Literal  # pragma: no cover

import functools
import inspect
import logging
import time
import warnings

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def scoped_file_logging_for_param(param_name: str = "log_file", level: int = logging.INFO):
    """Decorator: per-call isolated logging to the `log_file` parameter, then cleanup."""
    def _decorate(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Resolve arguments (supports both positional and keyword)
            bound = sig.bind_partial(*args, **kwargs)
            log_file = bound.arguments.get(param_name, None)

            # Use the same logger name as inside the function
            logger_name = f"{func.__module__}.{func.__name__}"
            logger = logging.getLogger(logger_name)

            # Reset previous handlers (from older calls)
            for h in list(logger.handlers):
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)

            # Configure for this call
            logger.setLevel(level)
            logger.propagate = False
            if log_file:
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
                logger.addHandler(fh)
            else:
                logger.addHandler(logging.NullHandler())

            try:
                return func(*args, **kwargs)
            finally:
                # Cleanup to avoid leaks and stale file locks
                for h in list(logger.handlers):
                    try:
                        h.flush()
                    except Exception:
                        pass
                    h.close()
                    logger.removeHandler(h)
        return _wrapper
    return _decorate


class RFCWithOOBProba(RandomForestClassifier):
    """
    RandomForestClassifier that, after fitting, recomputes `oob_score_`
    from OOB probabilities (`oob_decision_function_`) using a user-supplied
    callable, instead of the default label-based OOB score.

    oob_score_func
    --------------
    A user-supplied callable with signature ``f(y, proba) -> float`` assigned
    as an **instance attribute** *after* construction and *before* calling
    ``fit``. It is **not** a constructor argument by design (to keep sklearn
    compatibility with flat constructors and cloning). After ``fit``, the class sets:
        ``self.oob_score_ = float(oob_score_func(y, oob_decision_function_))``.

    Example
    -------
    >>> from sklearn.metrics import roc_auc_score
    >>> model = RFCWithOOBProba(bootstrap=True, oob_score=True, random_state=0)
    >>> model.oob_score_func = lambda y, proba: roc_auc_score(y, proba[:, 1])  # binary case
    >>> model.fit(X, y)
    >>> print(model.oob_score_)
    """

    def fit(self, X, y, *args, **kwargs):
        """Fit the forest; then recompute `oob_score_` from OOB probabilities (strict mode)."""
        # 1) Require the scorer to be set and callable
        if not hasattr(self, "oob_score_func") or not callable(self.oob_score_func):
            raise RuntimeError(
                "RFCWithOOBProba requires `model.oob_score_func = (y, proba) -> float` "
                "to be set BEFORE calling fit()."
            )

        out = super().fit(X, y, *args, **kwargs)

        # 2) Require OOB probabilities to exist (bootstrap=True and oob_score=True)
        if not hasattr(self, "oob_decision_function_"):
            raise RuntimeError(
                "OOB probabilities are unavailable after fit(). "
                "Ensure `bootstrap=True` and `oob_score=True` when constructing the estimator."
            )

        # 3) Strict recompute with helpful errors
        y_arr = np.asarray(y)
        proba = np.asarray(self.oob_decision_function_)
        try:
            self.oob_score_ = float(self.oob_score_func(y_arr, proba))
        except Exception as e:
            # Do NOT swallow the error: surface it with context
            raise RuntimeError(
                "Failed to recompute `oob_score_` from OOB probabilities. "
                "Check your `oob_score_func(y, proba)` implementation."
            ) from e

        return out


@scoped_file_logging_for_param("log_file", level=logging.INFO)
def tune_rf_oob(
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    # --- Search space ---
    max_features_grid: Sequence[object] = ("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0),
    max_depth_range: Tuple[int, int] = (4, 40),
    n_estimators_range: Tuple[int, int] = (50, 2000),
    min_samples_leaf_range: Tuple[int, int] = (1, 20),
    min_samples_split_range: Tuple[int, int] = (2, 40),
    tune_criterion: bool = True,
    # --- Pass to Random Forest class ---
    criterion: Optional[str] = None,
    class_weight: Optional[Union[str, Dict[str, float], List[Dict[str, float]]]] = None,
    # --- Optuna / runtime ---
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    n_trials: int = 40,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: int = 0,                 # 0 silent, 1 per-trial, 2 per-step (n_estimators value)
    # --- Logging ---
    log_file: Optional[str] = None,
    # --- Finalization / output ---
    refit: bool = True,
):
    """
    Classic Optuna HPO for RandomForest with an OOB-based objective.

    The study optimizes standard RF hyperparameters using TPE (by default).
    The objective is the OOB score computed by `score_func(y, y_pred_like)`.
    We treat any warnings during `fit`/`oob_score_` as errors and prune the trial.

    Parameters
    ----------
    X, y : array-like
        Training data. OOB requires `bootstrap=True` (enforced internally).
    problem : {'clf', 'reg'}
        Task type: classification or regression.
    score_func : callable
        OOB scoring function with signature ``score_func(y, y_pred_like) -> float``.
    greater_is_better : bool
        Direction of optimization for the study.

    max_features_grid : sequence, default=("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0)
        The set of values to sample for the `max_features` parameter of RandomForest.
    max_depth_range : tuple of int, default=(4, 40)
        The range of values to sample for the `max_depth` parameter of RandomForest.
    n_estimators_range : tuple of int, default=(50, 2000)
        The range of values to sample for the `n_estimators` parameter of RandomForest.
        Controls the number of trees in the forest. A higher number may increase performance but also training time.
    min_samples_leaf_range : tuple of int, default=(1, 20)
        The range of values to sample for the `min_samples_leaf` parameter of RandomForest.
    min_samples_split_range : tuple of int, default=(2, 40)
        The range of values to sample for the `min_samples_split` parameter of RandomForest.
    tune_criterion : bool, default=True
        If True, samples `criterion` from ['gini','entropy','log_loss'] (classification)
        or ['squared_error','absolute_error'] (regression).

    criterion : str, optional
        The criterion used for splitting (e.g., "gini" or "entropy" for classification).
        Must be `None` if `tune_criterion=True`. When `tune_criterion=False`, if `criterion`
        is None, the default criterion for the model (e.g., "gini" or "squared_error") will be used.
    class_weight : {'balanced', 'balanced_subsample'}, dict or list of dicts, optional
        Weights associated with classes in the form of a dictionary.
        - 'balanced' adjusts weights inversely proportional to class frequencies.
        - 'balanced_subsample' adjusts weights inversely proportional to class frequencies in each bootstrap sample.
        - A dictionary or a list of dictionaries can also be used to specify weights for each class explicitly.

    sampler : optuna.samplers.BaseSampler, optional
        The sampler used for Optuna trials.
    n_trials : int, default=40
        The number of trials (experiments) to run for hyperparameter tuning.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. Default is -1 (use all available cores).
    random_state : int, optional
        Random seed for reproducibility of results.
    verbose : int, default=0
        Verbosity level for logging. 0 for silent, 1 for per-trial logging, 2 for per-step logging.

    log_file : str, optional
        Path to the log file where the trial logs will be saved.
    refit : bool, default=True
        Whether to refit the best model after tuning.

    Returns
    -------
    final_model : Optional[RandomForestClassifier or RandomForestRegressor]
        Trained model if `refit=True`; otherwise `None`.
    study : optuna.Study
        The Optuna study with all trials.
    """

    # --- logging setup (INFO) ---
    func_name = inspect.currentframe().f_code.co_name  # type: ignore
    logger = logging.getLogger(f"{__name__}.{func_name}")

    assert callable(score_func), "score_func must be (y, y_pred_like) -> float"

    t0 = time.perf_counter()
    def _print(msg: str, v_gate: int = 1, step: Optional[int] = None) -> None:
        elapsed = time.perf_counter() - t0
        tagged = f"{msg} [t+{elapsed:.3f}s]" if step is None else f"[step {step}] {msg} [t+{elapsed:.3f}s]"
        if verbose >= v_gate:
            print(tagged)
        logger.info(tagged)

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize" if greater_is_better else "minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=0),
    )

    log_message = (
        f"Start tuning | problem={problem} | greater_is_better={greater_is_better} | "
     )
    log_message += f"max_features_grid={max_features_grid} | "
    log_message += f"max_depth_range={max_depth_range} | "
    log_message += f"n_estimators_range={n_estimators_range} | "
    log_message += f"min_samples_leaf_range={min_samples_leaf_range} | "
    log_message += f"min_samples_split_range={min_samples_split_range} | "
    log_message += f"tune_criterion={tune_criterion} "
    if criterion is not None:
        log_message += f"| criterion={criterion} "
    if class_weight is not None:
        log_message += f"| class_weight={class_weight} "
    if random_state is not None:
        log_message += f"| random_state={random_state} "
    _print(log_message, v_gate=1)

    def objective(trial: optuna.Trial) -> float:
        # Sample RF hyperparams (incl. n_estimators)
        params = dict(
            max_depth=trial.suggest_int("max_depth", *max_depth_range),
            max_features=trial.suggest_categorical("max_features", list(max_features_grid)),
            n_estimators=trial.suggest_int("n_estimators", *n_estimators_range),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *min_samples_leaf_range),
            min_samples_split=trial.suggest_int("min_samples_split", *min_samples_split_range),
        )

        if tune_criterion:
            if criterion is None:
                params["criterion"] = trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"] if problem == "clf" else ["squared_error", "absolute_error"],
                )
            else:
                raise ValueError("The criterion must be None when tune_criterion is True.")
        elif criterion is not None:
                params["criterion"] = criterion
        # Else default RF criterion is used

        if problem == 'reg' and class_weight is not None:
            raise ValueError("class_weight cannot be used for regression problems.")

        _print(f"[trial {trial.number}] params={params}", v_gate=1)

        if problem == 'clf':
            model = RFCWithOOBProba(
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight=class_weight,
                **params,
            )
            model.oob_score_func = score_func
        else:
            model = RandomForestRegressor(
                bootstrap=True,
                oob_score=score_func,
                n_jobs=n_jobs,
                random_state=random_state,
                **params,
            )

        n_trees = params['n_estimators']
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=Warning)
                model.fit(X, y)
                s = float(model.oob_score_)
        except Warning:
            _print(f"trees={n_trees:4d} | Model fit or OOB is incomplete!", v_gate=1, step=n_trees)
            raise optuna.TrialPruned()
        else:
            bad = not np.isfinite(s)
            if hasattr(model, "oob_decision_function_"):
                bad |= not np.isfinite(model.oob_decision_function_).all()
            if hasattr(model, "oob_prediction_"):
                bad |= not np.isfinite(model.oob_prediction_).all()
            if bad:
                _print(f"trees={n_trees:4d} | Invalid OOB arrays/score", v_gate=1, step=n_trees)
                raise optuna.TrialPruned()

            _print(f"trees={n_trees:4d} | oob_score={s:.6f}", v_gate=1, step=n_trees)
            trial.report(s, step=n_trees)
            if trial.should_prune():
                _print(f"pruned at trees={n_trees}", v_gate=1, step=n_trees)
                raise optuna.TrialPruned()

        return s

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    completed_trials = [t.state == optuna.trial.TrialState.COMPLETE and t.value is not None for t in study.trials]
    if len(completed_trials) == 0:
        _print(f"No completed trials.", v_gate=1)
        return None, study

    best_trial = study.best_trial
    _print(f"BEST trial={best_trial.number} | score={best_trial.value:.6f}", v_gate=1)
    _print(f"BEST params={best_trial.params}", v_gate=1)

    if refit:
        final_model = (
            RandomForestClassifier if problem == "clf" else RandomForestRegressor
        )(
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
            **dict(best_trial.params),
        ).fit(X, y)
    else:
        final_model = None  # no refit; best hyperparameters are in study.best_trial.params

    return final_model, study


@scoped_file_logging_for_param("log_file", level=logging.INFO)
def tune_rf_oob_plateau(
    X: np.ndarray,
    y: np.ndarray,
    problem: Literal["clf", "reg"],
    score_func: Callable[[np.ndarray, np.ndarray], float],
    greater_is_better: bool,
    *,
    # --- Search space ---
    max_features_grid: Sequence[object] = ("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0),
    max_depth_range: Tuple[int, int] = (4, 40),
    min_samples_leaf_range: Tuple[int, int] = (1, 20),
    min_samples_split_range: Tuple[int, int] = (2, 40),
    tune_criterion: bool = True,

    # --- Pass to Random Forest class ---
    criterion: Optional[str] = None,
    class_weight: Optional[Union[str, Dict[str, float], List[Dict[str, float]]]] = None,

    # --- n_estimators triplet mechanics ---
    n_estimators_start: int = 100,
    scale_factor: float = 1.5,
    delta: float = 1e-3,
    max_trees: int = 5000,

    # --- Optuna / runtime ---
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    n_trials: int = 40,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: int = 0,

    # --- Logging ---
    log_file: Optional[str] = None,

    # --- Finalization / output ---
    refit: bool = True,
):
    """
    Random Forest tuning with OOB-based plateau search for `n_estimators` + Optuna HPO (triplet-based).

    This routine optimizes standard RandomForest hyperparameters with Optuna (TPE by default)
    and finds a near-minimal tree count via a geometric **triplet** of sizes T = [L, B, R]:
      - L (left probe): may be invalid; never reported to Optuna.
      - B (baseline): objective value of the trial (OOB score at B).
      - R (right neighbor): used to confirm a right-side plateau.

    Plateau rule (eps=1e-12):
      • If B or R is invalid -> no plateau.
      • Otherwise, plateau := |s(R) - s(B)| / max(|s(B)|, eps) ≤ delta.
        Additionally, left_close := L is valid and |s(L) - s(B)| / max(|s(B)|, eps) ≤ delta.

    Shift policy (triplet sliding; geometric spacing by `scale_factor`):
      • plateau & left_close     -> shift LEFT  (try to reduce trees)
      • plateau & not left_close -> STAY
      • no plateau               -> shift RIGHT 

    Pruning policy:
      - During growth over [L, B, R], pruning is allowed only **after** a plateau is observed (when reporting s(R)),
        to avoid aborting configurations just because trees are too few.
      - If a trial yields **no plateau**, it is explicitly pruned (so TPE won’t learn from it).

    Revisit phase:
      After the initial `study.optimize`, the routine repeatedly enqueues the current best hyperparameters
      and runs single-trial updates while the selected baseline `B` keeps decreasing. The loop terminates when
      the baseline cannot be reduced further or upon reaching the revisit limit.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    problem : {'clf', 'reg'}
        Task type: classification ('clf') or regression ('reg').
    score_func : callable
        OOB scoring function with signature ``score_func(y, y_pred_like) -> float``.

        - Classification: ``y_pred_like`` is the forest’s ``oob_decision_function_``
          (array of shape (n_samples, n_classes) with class probabilities).
          Example (binary ROC AUC): ``score_func=lambda y, proba: roc_auc_score(y, proba[:, 1])``.
          Use ``greater_is_better=True``.

        - Regression: ``y_pred_like`` is the forest’s ``oob_prediction_``
          (array of shape (n_samples,) or (n_samples, n_outputs)).
          Example (MSE): ``score_func=mean_squared_error``.
          Use ``greater_is_better=False``.

        The function must be deterministic and side-effect free.
    greater_is_better : bool
        Direction of optimization (True for metrics to maximize like ROC AUC; False for those to minimize).

    max_features_grid : sequence of {'sqrt' or float in (0, 1]}, default=("sqrt", 0.25, 1/3, 0.5, 0.7, 1.0)
        Candidate values for `max_features`. 'sqrt' is passed as-is; floats are interpreted as fractions of `n_features`.
    max_depth_range : (int, int), default=(4, 40)
        Inclusive sampling range for `max_depth`.
    min_samples_leaf_range : (int, int), default=(1, 20)
        Inclusive sampling range for `min_samples_leaf`.
    min_samples_split_range : (int, int), default=(2, 40)
        Inclusive sampling range for `min_samples_split`.
    tune_criterion : bool, default=True
        If True, samples `criterion` from ['gini','entropy','log_loss'] (classification)
        or ['squared_error','absolute_error'] (regression).

    criterion : str, optional
        The criterion used for splitting (e.g., "gini" or "entropy" for classification).
        Must be `None` if `tune_criterion=True`. When `tune_criterion=False`, if `criterion`
        is None, the default criterion for the model (e.g., "gini" or "squared_error") will be used.
    class_weight : {'balanced', 'balanced_subsample'}, dict or list of dicts, optional
        Weights associated with classes in the form of a dictionary.
        - 'balanced' adjusts weights inversely proportional to class frequencies.
        - 'balanced_subsample' adjusts weights inversely proportional to class frequencies in each bootstrap sample.
        - A dictionary or a list of dictionaries can also be used to specify weights for each class explicitly.

    n_estimators_start : int, default=100
        Baseline initializer for the first triplet ([L, B, R] is built around this baseline).
    scale_factor : float, default=1.5
        Geometric spacing (> 1) between the triplet points.
    delta : float, default=2e-3
        Relative tolerance in plateau checks: `abs(a-b)/max(|b|, eps) <= delta`.
    max_trees : int, default=5000
        Upper bound for feasible tree counts.

    sampler : optuna.samplers.BaseSampler or None, default=None
        Optuna sampler. If None, uses TPESampler(seed=random_state).
    n_trials : int, default=40
        Number of Optuna trials in the initial optimization phase.
    n_jobs : int, default=-1
        Parallel workers for the RandomForest.
    random_state : int or None, default=None
        Seed forwarded to the RandomForest and, if TPESampler is used, to the sampler.
    verbose : int, default=0
        Console verbosity: 0 = silent, 1 = per-trial, 2 = per-step (triplet points).
        File logging follows the same gates.

    log_file : str or None, default=None
        If provided, write INFO-level logs to this file.

    refit : bool, default=True
        If True, fit a final RandomForest on (X, y) with the best hyperparameters and
        `n_estimators = B` from the selected trial.

    Returns
    -------
    final_model : Optional[RandomForestClassifier or RandomForestRegressor]
        Trained model if `refit=True`; otherwise `None`.
    best_n_estimators : Optional[int]
        The baseline `B` from the selected trial (i.e., chosen number of trees).
    study : optuna.Study
        The Optuna study with all trials. Per-trial metadata (triplet and scores) are in `user_attrs`.
    plateau_reached : bool
        Whether at least one COMPLETE plateau trial was observed in the optimization (incl. revisit).

    Notes
    -----
    - **OOB requirement**: `bootstrap=True` is enforced to enable OOB.
    - *Shift coding:* we store the triplet shift as an integer ``shift ∈ {-1, 0, +1} ≡ {left, stay, right}`` in ``user_attrs``.
    - Any `Warning` during fit/OOB is treated as an error for that point, and the score is considered invalid.
    - Triplets are validated (strict monotonicity and bounds); we do **not** clamp to [1, max_trees].
    - If a shift produces an invalid triplet, the triplet stays unchanged and the shift is marked as blocked.
    """

    # --- logging setup (INFO) ---
    func_name = inspect.currentframe().f_code.co_name
    logger = logging.getLogger(f"{__name__}.{func_name}")

    # --- checks ---
    assert callable(score_func), "score_func must be (y, y_pred_like) -> float"
    assert isinstance(n_estimators_start, int) and n_estimators_start >= 2, "n_estimators_start must be int >= 2"
    assert scale_factor > 1.0, "scale_factor must be > 1"
    assert isinstance(max_trees, int) and max_trees >= 3, "max_trees must be int >= 3"

    t0 = time.perf_counter()
    def _print(msg: str, v_gate: int = 1, step: Optional[int] = None):
        elapsed = time.perf_counter() - t0
        tagged = f"{msg} [t+{elapsed:.3f}s]" if step is None else f"[step {step}] {msg} [t+{elapsed:.3f}s]"
        if verbose >= v_gate:
            print(tagged)
        logger.info(tagged)

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction = "maximize" if greater_is_better else "minimize", 
        sampler = sampler,
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)
    )

    def _valid_triplet(triplet: Tuple[int, int, int], raise_on_error: bool = True) -> bool:
        """Validate [L, B, R] for strict monotonicity and bounds."""

        L, B, R = triplet
        if 1 <= L < B < R <= max_trees:
            return True
        if not raise_on_error:
            return False
        raise ValueError("The triplet either reaches bounds or is too small.")

    def _make_triplet_from_baseline(n_estimators_base: int) -> Tuple[int, int, int]:
        """
        Construct a geometric triplet [L, B, R] around the given baseline.
        Pure geometric construction (no clamping).
        """

        L = int(round(n_estimators_base / scale_factor))
        B = n_estimators_base
        R = int(round(n_estimators_base * scale_factor))

        _valid_triplet((L, B, R), raise_on_error=True)
        return L, B, R

    def _slide_triplet(triplet: Tuple[int, int, int], shift: int) -> Tuple[Tuple[int, int, int], bool]:
        """
        Slide the triplet one step left (shift = -1), stay (0), or right (+1).
        If the proposed triplet_nxt is invalid, returns the original triplet and False.
        """

        L, B, R = triplet
        if shift == -1:
            B, R = L, B
            L = int(round(B / scale_factor))
        if shift == 1:
            L, B = B, R
            R = int(round(B * scale_factor))

        shift_status = _valid_triplet((L, B, R), raise_on_error=False)
        return (L, B, R) if shift_status else triplet, shift_status

    triplet = _make_triplet_from_baseline(n_estimators_start)

    log_message = (
        f"Start tuning | problem={problem} | greater_is_better={greater_is_better} | "
        f"delta={delta:.3g} | scale_factor={scale_factor:.3g} | triplet0={triplet} | "
    )
    log_message += f"max_features_grid={max_features_grid} | "
    log_message += f"max_depth_range={max_depth_range} | "
    log_message += f"min_samples_leaf_range={min_samples_leaf_range} | "
    log_message += f"min_samples_split_range={min_samples_split_range} | "
    log_message += f"tune_criterion={tune_criterion} "
    if criterion is not None:
        log_message += f"| criterion={criterion} "
    if class_weight is not None:
        log_message += f"| class_weight={class_weight} "
    if random_state is not None:
        log_message += f"| random_state={random_state} "
    _print(log_message, v_gate=1)

    def objective(trial: optuna.Trial) -> float:
        nonlocal triplet
        trial.set_user_attr("triplet", triplet)

        # Sample RF hyperparams
        params = dict(
            max_depth=trial.suggest_int("max_depth", *max_depth_range),
            max_features=trial.suggest_categorical("max_features", list(max_features_grid)),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *min_samples_leaf_range),
            min_samples_split=trial.suggest_int("min_samples_split", *min_samples_split_range),
        )

        if tune_criterion:
            if criterion is None:
                params["criterion"] = trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"] if problem == "clf" else ["squared_error", "absolute_error"],
                )
            else:
                raise ValueError("The criterion must be None when tune_criterion is True.")
        elif criterion is not None:
                params["criterion"] = criterion
        # Else default RF criterion is used

        if problem == 'reg' and class_weight is not None:
            raise ValueError("class_weight cannot be used for regression problems.")

        _print(f"[trial {trial.number}] params={params} | triplet={triplet}", v_gate=1)

        if problem == 'clf':
            model = RFCWithOOBProba(
                n_estimators=0,
                warm_start=True,
                bootstrap=True,
                oob_score=True,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight=class_weight,
                **params,
            )
            model.oob_score_func = score_func
        else:
            model = RandomForestRegressor(
                n_estimators=0,
                warm_start=True,
                bootstrap=True,
                oob_score=score_func,
                n_jobs=n_jobs,
                random_state=random_state,
                **params,
            )

        scores: List[Optional[float]] = []
        eps = 1e-12
        left_close: bool = False
        plateau: bool = False  # "right-side plateau" flag based on scores[1:]

        # Grow across the triplet; DO NOT report/prune at j == 0 (left probe)
        for j, m in enumerate(triplet):
            model.n_estimators = m
            
            try:
                # Turn ANY warnings into errors just for this block
                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=Warning)
                    model.fit(X, y)
                    s_j = float(model.oob_score_)  # may emit UserWarning when OOB is incomplete
            except Warning as w:
                _print(f"trees={m:4d} | Model fit or OOB is incomplete!", v_gate=2, step=m)
                s_j = None
            else:
                bad = not np.isfinite(s_j)
                if hasattr(model, "oob_decision_function_"):
                    bad |= not np.isfinite(model.oob_decision_function_).all()
                if hasattr(model, "oob_prediction_"):
                    bad |= not np.isfinite(model.oob_prediction_).all()
                if bad:
                    s_j = None
                    _print(f"trees={m:4d} | Invalid OOB arrays/score", v_gate=2, step=m)
                else:
                    _print(f"trees={m:4d} | oob_score={s_j:.6f}", v_gate=2, step=m)
            scores.append(s_j)

            if j == 0:
                continue
            else:
                if scores[j] is None:
                    plateau = False
                    break
                if j == 1:
                    left_close = (
                        scores[0] is not None 
                        and abs(scores[0] - scores[1]) / max(abs(scores[1]), eps) <= delta
                    )
                else:
                    # j == 2: check right-side plateau
                    plateau = abs(scores[2] - scores[1]) / max(abs(scores[1]), eps) <= delta
                    if plateau:
                        # allow pruning ONLY when plateau is already observed on the right
                        trial.report(s_j, step=m)
                        if trial.should_prune():
                            _print(f"pruned at trees={m}", v_gate=1, step=m)
                            raise optuna.TrialPruned()

        def _register_shift():
            trial.set_user_attr("shift", shift)
            trial.set_user_attr("shift_status", shift_status)
            trial.set_user_attr("triplet_nxt", triplet)
            _print(f"shift={trial.user_attrs['shift']}", v_gate=1)
            _print(f"shift_status={trial.user_attrs['shift_status']}", v_gate=1)

        # Register trial only if plateau; otherwise prune the whole trial
        if plateau:
            trial.set_user_attr("scores", scores)
            _print(f"score(n_estimators={triplet[1]})={scores[1]:.6f}", v_gate=1)

            if left_close:
                # plateau and left_close: move triplet to the left
                shift = -1
                triplet, shift_status = _slide_triplet(triplet, shift)
            else:
                # plateau and no left_close: do not move triplet
                shift = 0
                shift_status = None

            _register_shift()
        else:
            # no plateau: shift triplet to the right and prune the trial so TPE doesn't learn from it
            shift = 1
            triplet, shift_status = _slide_triplet(triplet, shift)
            _register_shift()
            raise optuna.TrialPruned()

        return scores[1]

    # Reduce Optuna console noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optimize
    study.optimize(objective, n_trials=n_trials)

    def _is_plateau_trial(trial: optuna.trial.FrozenTrial) -> bool:
        """
        Return True iff the trial is COMPLETE and contains required user_attrs
        for a plateau trial (scores present, shift != 1, has next triplet metadata).
        """

        # trial is not COMPLETE
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return False

        # missing attribute
        for attr in ("triplet", "scores", "shift", "triplet_nxt"):
            if attr not in trial.user_attrs:
                return False

        # shift == 1 (no plateau)
        if trial.user_attrs["shift"] == 1:
            return False

        return True

    def _print_best_trial(trial: optuna.trial.FrozenTrial, v_gate: int = 1):
        _print(f"BEST trial={trial.number} | best_n_estimators={trial.user_attrs['triplet'][1]} | score={trial.value:.6f}", v_gate)
        _print(f"BEST params={trial.params}", v_gate)
        _print(f"BEST shift={trial.user_attrs['shift']}", v_gate)
        _print(f"BEST shift_status={trial.user_attrs['shift_status']}", v_gate)
        _print(f"BEST triplet={trial.user_attrs['triplet']}", v_gate)
        _print(f"BEST triplet_nxt={trial.user_attrs['triplet_nxt']}", v_gate)
        scores = trial.user_attrs['scores']
        score_str = [None if s is None else f"{float(s):.6f}" for s in scores]
        _print(f"BEST scores={score_str}", v_gate)

    completed_trials = [t.state == optuna.trial.TrialState.COMPLETE and t.value is not None for t in study.trials]
    if len(completed_trials) == 0:
        _print(f"No completed trials.", v_gate=1)
        return None, None, study, False

    best_trial = study.best_trial
    if _is_plateau_trial(best_trial):
        _print_best_trial(best_trial, v_gate=1)
        # Go to revisit phase
    else:
        _print(f"No plateau reached.", v_gate=1)
        if "triplet" in best_trial.user_attrs:
            best_n_estimators = best_trial.user_attrs["triplet"][1]
        else:
            best_n_estimators = None
        return None, best_n_estimators, study, False

    # Revisit phase: repeat the current best hyperparameters while triplet moves left
    triplet = best_trial.user_attrs["triplet_nxt"] # correct triplet after best trial
    best_n_estimators = best_trial.user_attrs["triplet"][1]
    for k in range(n_trials):
        if best_trial.user_attrs["shift_status"]:
            
            study.enqueue_trial(best_trial.params)
            study.optimize(objective, n_trials=1)
            
            nxt_trial = study.trials[-1]
            if _is_plateau_trial(nxt_trial):
                nxt_n_estimators = nxt_trial.user_attrs["triplet"][1]
                assert nxt_n_estimators < best_n_estimators
                _print(f"[revisit {k+1}] n_estimators reduced from {best_n_estimators} to {nxt_n_estimators}", v_gate=1)
                best_trial = nxt_trial
                best_n_estimators = best_trial.user_attrs["triplet"][1]
            else:
                _print(f"[revisit {k+1}] n_estimators cannot be changed from {best_n_estimators}", v_gate=1)
                break
        else:
            _print(f"[revisit {k+1}] n_estimators lower limit reached: {best_n_estimators}", v_gate=1)
            break

    _print_best_trial(best_trial, v_gate=1)

    # Conditional final refit
    if refit:
        final_model = (
            RandomForestClassifier if problem == "clf" else RandomForestRegressor
        )(
            n_estimators=best_n_estimators,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=random_state,
            **dict(best_trial.params),
        ).fit(X, y)
    else:
        final_model = None  # no refit; best hyperparameters are in study.best_trial.params

    return final_model, best_n_estimators, study, True
