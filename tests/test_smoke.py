"""Smoke tests for rf_plateau_hpo.

Scope: verify that the three public tuners import, run end to end, and return
what their docstrings promise. These tests exist so that a dependency bump
(numpy, scikit-learn, Optuna) cannot silently break the public API. They are
not a substitute for notebooks/paper_repro.ipynb.

Design notes
------------
* The tuners are stochastic and prune aggressively: ``tune_rf_oob_plateau``
  prunes every trial that does not reach a plateau, so a run can legitimately
  end with zero completed trials and return ``(None, None, study, False)``.
  Assertions below are written to accept that outcome rather than to fail on
  an unlucky seed. What they pin down is the *contract*: arity of the return
  tuple, types, and the defining property that the plateau tuner never puts
  ``n_estimators`` into the sampled search space.
* Search spaces are deliberately narrow and forests tiny so the whole file
  runs in seconds on a CI runner.
"""

import inspect

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from rf_plateau_hpo.core import (
    RFCWithOOBProba,
    tune_rf_oob,
    tune_rf_oob_bohb,
    tune_rf_oob_plateau,
)


def auc_binary(y_true, proba):
    """OOB scorer for binary classification: takes (y, oob_decision_function_)."""
    return roc_auc_score(y_true, proba[:, 1])


# A small, well-separated problem. Enough samples that out-of-bag coverage is
# complete even for the smallest forests used here, which keeps trials from
# being pruned for reasons unrelated to what is being tested.
@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=0,
    )
    return X, y


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(
        n_samples=400,
        n_features=10,
        n_informative=5,
        noise=0.3,
        random_state=0,
    )
    return X, y


# Shared narrow search space, applied to every tuner so runs stay fast and the
# sampled configurations are all trainable.
FAST_SPACE = dict(
    max_features_grid=("sqrt",),
    max_depth_range=(3, 8),
    min_samples_leaf_range=(1, 5),
    min_samples_split_range=(2, 10),
    tune_criterion=False,
)


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------

def test_public_tuners_are_callable():
    assert callable(tune_rf_oob)
    assert callable(tune_rf_oob_bohb)
    assert callable(tune_rf_oob_plateau)


def test_dataset_loader_is_importable():
    from rf_plateau_hpo.datasets.dataloader import load_dataset

    assert callable(load_dataset)


@pytest.mark.parametrize(
    "func", [tune_rf_oob, tune_rf_oob_bohb, tune_rf_oob_plateau]
)
def test_tuner_signatures_are_stable(func):
    """The five leading arguments are positional; everything else is keyword-only.

    README examples and the reproducibility notebook both rely on this, so a
    silent reordering would break user code without breaking anything else.
    """
    params = list(inspect.signature(func).parameters.values())
    leading = [p.name for p in params[:5]]
    assert leading == ["X", "y", "problem", "score_func", "greater_is_better"]

    keyword_only = {
        p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    for name in ("n_trials", "random_state", "n_jobs", "verbose", "refit"):
        assert name in keyword_only


def test_plateau_tuner_exposes_its_knobs():
    """delta, scale_factor and max_trees are the method's user-facing controls."""
    params = inspect.signature(tune_rf_oob_plateau).parameters
    for name in ("n_estimators_start", "scale_factor", "delta", "max_trees"):
        assert name in params

    assert "n_estimators_range" not in params
    assert "n_estimators_ladder" not in params


# --------------------------------------------------------------------------
# RFCWithOOBProba
# --------------------------------------------------------------------------

def test_oob_wrapper_requires_scorer_before_fit(clf_data):
    """Documented strict behaviour: fit() without oob_score_func must raise."""
    X, y = clf_data
    model = RFCWithOOBProba(
        n_estimators=20, bootstrap=True, oob_score=True, random_state=0
    )
    with pytest.raises(RuntimeError, match="oob_score_func"):
        model.fit(X, y)


def test_oob_wrapper_recomputes_score_from_probabilities(clf_data):
    X, y = clf_data
    model = RFCWithOOBProba(
        n_estimators=50, bootstrap=True, oob_score=True, random_state=0, n_jobs=1
    )
    model.oob_score_func = auc_binary
    model.fit(X, y)

    assert hasattr(model, "oob_decision_function_")
    expected = auc_binary(y, model.oob_decision_function_)
    assert model.oob_score_ == pytest.approx(expected)
    # An AUC, not the default accuracy-style OOB score.
    assert 0.5 < model.oob_score_ <= 1.0


# --------------------------------------------------------------------------
# tune_rf_oob — fixed-range TPE baseline
# --------------------------------------------------------------------------

def test_tune_rf_oob_runs_and_samples_n_estimators(clf_data):
    X, y = clf_data
    model, study = tune_rf_oob(
        X,
        y,
        "clf",
        auc_binary,
        True,
        n_estimators_range=(20, 60),
        n_trials=3,
        random_state=0,
        n_jobs=1,
        verbose=0,
        **FAST_SPACE,
    )

    assert len(study.trials) == 3

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        pytest.skip("all trials pruned; nothing to assert about the best trial")

    assert isinstance(model, RandomForestClassifier)
    # The baseline's defining property: the tree count IS a sampled parameter,
    # and it stays inside the range it was given.
    assert "n_estimators" in study.best_params
    assert 20 <= study.best_params["n_estimators"] <= 60


def test_tune_rf_oob_regression(reg_data):
    """Exercises the callable-oob_score path, which needs scikit-learn >= 1.3.

    This is the test most likely to catch an incompatible scikit-learn after a
    dependency bump, since the regressor passes score_func straight through to
    the estimator instead of using the classifier wrapper.
    """
    X, y = reg_data
    model, study = tune_rf_oob(
        X,
        y,
        "reg",
        mean_squared_error,
        False,
        n_estimators_range=(20, 40),
        n_trials=2,
        random_state=0,
        n_jobs=1,
        verbose=0,
        **FAST_SPACE,
    )

    assert study.direction.name == "MINIMIZE"
    completed = [t for t in study.trials if t.value is not None]
    if completed:
        assert isinstance(model, RandomForestRegressor)
        assert study.best_value >= 0.0


# --------------------------------------------------------------------------
# tune_rf_oob_bohb — Hyperband-style baseline
# --------------------------------------------------------------------------

def test_tune_rf_oob_bohb_returns_four_values(clf_data):
    X, y = clf_data
    ladder = (20, 40, 80)
    model, best_n, study, stopped = tune_rf_oob_bohb(
        X,
        y,
        "clf",
        auc_binary,
        True,
        n_estimators_ladder=ladder,
        hyperband_reduction_factor=3,
        n_trials=3,
        random_state=0,
        n_jobs=1,
        verbose=0,
        **FAST_SPACE,
    )

    assert len(study.trials) == 3
    assert isinstance(stopped, bool)

    if best_n is None:
        pytest.skip("no completed trials in this run")

    # The resource is the ladder, so the selected count must come from it.
    assert best_n in ladder
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == best_n
    # n_estimators is the budget, never a sampled hyperparameter.
    assert "n_estimators" not in study.best_params


def test_tune_rf_oob_bohb_rejects_non_increasing_ladder(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError, match="strictly increasing"):
        tune_rf_oob_bohb(
            X,
            y,
            "clf",
            auc_binary,
            True,
            n_estimators_ladder=(40, 20),
            n_trials=1,
            random_state=0,
            n_jobs=1,
            verbose=0,
            **FAST_SPACE,
        )


# --------------------------------------------------------------------------
# tune_rf_oob_plateau — the method
# --------------------------------------------------------------------------

def test_tune_rf_oob_plateau_returns_four_values(clf_data):
    X, y = clf_data
    model, best_n, study, plateau_found = tune_rf_oob_plateau(
        X,
        y,
        "clf",
        auc_binary,
        True,
        n_estimators_start=32,
        scale_factor=2.0,
        delta=5e-2,          # loose tolerance: a plateau is reachable in a few trials
        max_trees=4000,
        n_trials=4,
        random_state=0,
        n_jobs=1,
        verbose=0,
        **FAST_SPACE,
    )

    assert isinstance(plateau_found, bool)
    # The revisit phase can append trials, so this is a lower bound.
    assert len(study.trials) >= 4

    # THE defining property of the method: the tree count is never sampled.
    for trial in study.trials:
        assert "n_estimators" not in trial.params

    if not plateau_found:
        # Documented outcome: increase n_trials or max_trees. best_n may still
        # carry the last baseline the search reached.
        assert best_n is None or best_n > 0
        return

    assert isinstance(best_n, int)
    assert 0 < best_n <= 4000
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == best_n


def test_tune_rf_oob_plateau_records_triplet_metadata(clf_data):
    """Downstream analysis in notebooks/ reads these user_attrs by name."""
    X, y = clf_data
    _, _, study, _ = tune_rf_oob_plateau(
        X,
        y,
        "clf",
        auc_binary,
        True,
        n_estimators_start=32,
        scale_factor=2.0,
        delta=5e-2,
        max_trees=4000,
        n_trials=3,
        random_state=1,
        n_jobs=1,
        verbose=0,
        refit=False,
        **FAST_SPACE,
    )

    for trial in study.trials:
        attrs = trial.user_attrs
        assert "triplet" in attrs
        L, B, R = attrs["triplet"]
        assert 1 <= L < B < R, "triplet must be strictly increasing"
        assert "trees_built" in attrs
        assert "shift" in attrs
        assert attrs["shift"] in (-1, 0, 1)


def test_tune_rf_oob_plateau_validates_its_parameters(clf_data):
    X, y = clf_data
    common = dict(
        n_trials=1, random_state=0, n_jobs=1, verbose=0, **FAST_SPACE
    )

    with pytest.raises(AssertionError, match="scale_factor"):
        tune_rf_oob_plateau(
            X, y, "clf", auc_binary, True, scale_factor=1.0, **common
        )

    with pytest.raises(AssertionError, match="n_estimators_start"):
        tune_rf_oob_plateau(
            X, y, "clf", auc_binary, True, n_estimators_start=1, **common
        )

    with pytest.raises(AssertionError, match="max_trees"):
        tune_rf_oob_plateau(
            X, y, "clf", auc_binary, True, max_trees=2, **common
        )


def test_tune_rf_oob_plateau_rejects_class_weight_for_regression(reg_data):
    X, y = reg_data
    with pytest.raises(ValueError, match="class_weight"):
        tune_rf_oob_plateau(
            X,
            y,
            "reg",
            mean_squared_error,
            False,
            class_weight="balanced",
            n_estimators_start=32,
            scale_factor=2.0,
            n_trials=1,
            random_state=0,
            n_jobs=1,
            verbose=0,
            **FAST_SPACE,
        )


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_same_seed_gives_same_result(clf_data):
    X, y = clf_data
    kwargs = dict(
        n_estimators_range=(20, 40),
        n_trials=3,
        random_state=7,
        n_jobs=1,
        verbose=0,
        refit=False,
        **FAST_SPACE,
    )

    _, study_a = tune_rf_oob(X, y, "clf", auc_binary, True, **kwargs)
    _, study_b = tune_rf_oob(X, y, "clf", auc_binary, True, **kwargs)

    values_a = [t.value for t in study_a.trials]
    values_b = [t.value for t in study_b.trials]
    assert values_a == values_b

    params_a = [t.params for t in study_a.trials]
    params_b = [t.params for t in study_b.trials]
    assert params_a == params_b
