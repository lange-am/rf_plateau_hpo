"""Behavioural tests for the tuning algorithms in rf_plateau_hpo.core.

Where test_smoke.py checks the shape of the public API, this file checks that
the algorithms actually do what the paper describes. The technique throughout
is to drive the tolerance parameter to an extreme where the correct behaviour
is known in advance, independently of the particular out-of-bag scores a run
happens to produce:

* ``delta`` close to zero  -> no triplet can ever satisfy the plateau
  criterion, so every plateau trial must be pruned and the triplet must walk
  to the right.
* ``delta`` close to one   -> every triplet satisfies it, including on the
  left, so the triplet must walk to the left and settle at the lower bound.

The same lever applies to the early-stopping rule inside tune_rf_oob_bohb.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def auc_binary(y_true, proba):
    return roc_auc_score(y_true, proba[:, 1])


def auc_multiclass(y_true, proba):
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


FAST_SPACE = dict(
    max_features_grid=("sqrt",),
    max_depth_range=(3, 8),
    min_samples_leaf_range=(1, 5),
    min_samples_split_range=(2, 10),
    tune_criterion=False,
)


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=5, n_redundant=2, random_state=0
    )
    return X, y


# --------------------------------------------------------------------------
# Plateau search: the triplet must move in the documented direction
# --------------------------------------------------------------------------

def test_no_plateau_is_reachable_when_tolerance_is_zero(clf_data):
    """delta ~ 0: the plateau criterion can never hold.

    Documented consequences: no trial completes, the function reports that no
    plateau was found, every trial is pruned with reason "no_plateau", and the
    triplet shifts right each time.
    """
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    model, best_n, study, plateau_found = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=32, scale_factor=2.0,
        delta=1e-15, max_trees=100000,
        n_trials=4, random_state=0, n_jobs=1, verbose=0,
        **FAST_SPACE,
    )

    assert plateau_found is False
    assert model is None

    shifts = [t.user_attrs["shift"] for t in study.trials]
    assert shifts == [1] * len(shifts), "every trial must shift right"

    reasons = [t.user_attrs.get("pruned") for t in study.trials]
    assert all(r == "no_plateau" for r in reasons)

    # The baseline must grow monotonically as the triplet walks right.
    baselines = [t.user_attrs["triplet"][1] for t in study.trials]
    assert baselines == sorted(baselines)
    assert baselines[-1] > baselines[0]


def test_plateau_walks_left_when_tolerance_is_loose(clf_data):
    """delta ~ 1: every comparison is within tolerance on both sides.

    The documented rule is "plateau and left_close -> shift left", so the
    baseline must shrink until the triplet can no longer be slid.
    """
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    model, best_n, study, plateau_found = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=256, scale_factor=2.0,
        delta=0.9, max_trees=100000,
        n_trials=8, random_state=0, n_jobs=1, verbose=0,
        **FAST_SPACE,
    )

    assert plateau_found is True
    assert isinstance(best_n, int) and best_n > 0

    baselines = [t.user_attrs["triplet"][1] for t in study.trials]
    assert baselines[-1] < baselines[0], "the triplet must move towards fewer trees"

    # Selected count is strictly smaller than the starting point: the whole
    # purpose of the method is to come back with fewer trees than requested.
    assert best_n < 256
    assert model.n_estimators == best_n


def test_triplet_geometry_is_preserved_across_trials(clf_data):
    """Every triplet is (round(B/sf), B, round(B*sf)) and strictly increasing."""
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    sf = 2.0
    _, _, study, _ = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=64, scale_factor=sf,
        delta=1e-2, max_trees=100000,
        n_trials=5, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    for trial in study.trials:
        L, B, R = trial.user_attrs["triplet"]
        assert 1 <= L < B < R
        assert L == int(round(B / sf))
        assert R == int(round(B * sf))


def test_trees_built_never_exceeds_the_triplet_maximum(clf_data):
    """trees_built is the cost metric reported in the paper; it must be real."""
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    _, _, study, _ = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=64, scale_factor=2.0,
        delta=1e-2, max_trees=100000,
        n_trials=4, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    for trial in study.trials:
        L, B, R = trial.user_attrs["triplet"]
        built = trial.user_attrs["trees_built"]
        assert 0 <= built <= R


def test_max_trees_bounds_the_search(clf_data):
    """With an unreachable plateau, the walk right must stop at max_trees."""
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    max_trees = 300
    _, _, study, plateau_found = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=32, scale_factor=2.0,
        delta=1e-15, max_trees=max_trees,
        n_trials=6, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    assert plateau_found is False
    for trial in study.trials:
        L, B, R = trial.user_attrs["triplet"]
        assert R <= max_trees, "the triplet must never exceed max_trees"

    # Once the bound is hit the slide is refused and reported as blocked.
    blocked = [t.user_attrs["shift_status"] for t in study.trials]
    assert blocked[-1] is False


# --------------------------------------------------------------------------
# Hyperband / early-stopping baseline
# --------------------------------------------------------------------------

def test_bohb_without_early_stopping_uses_the_whole_ladder(clf_data):
    """delta < 0 disables the stopping rule, so the top rung must be selected."""
    from rf_plateau_hpo.core import tune_rf_oob_bohb

    X, y = clf_data
    ladder = (20, 40, 80)
    _, best_n, study, stopped = tune_rf_oob_bohb(
        X, y, "clf", auc_binary, True,
        n_estimators_ladder=ladder, hyperband_reduction_factor=1,
        delta=-1.0, n_trials=3, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    assert stopped is False
    assert best_n == ladder[-1]


def test_bohb_with_loose_tolerance_stops_at_the_first_rung(clf_data):
    """delta ~ 1 makes the first comparison succeed, selecting the left point."""
    from rf_plateau_hpo.core import tune_rf_oob_bohb

    X, y = clf_data
    ladder = (20, 40, 80)
    _, best_n, study, stopped = tune_rf_oob_bohb(
        X, y, "clf", auc_binary, True,
        n_estimators_ladder=ladder, hyperband_reduction_factor=1,
        delta=0.9, n_trials=3, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    assert stopped is True
    assert best_n == ladder[0], "stopping at rung j selects the left point"


def test_bohb_reports_every_rung_it_evaluates(clf_data):
    from rf_plateau_hpo.core import tune_rf_oob_bohb

    X, y = clf_data
    ladder = (20, 40, 80)
    _, _, study, _ = tune_rf_oob_bohb(
        X, y, "clf", auc_binary, True,
        n_estimators_ladder=ladder, hyperband_reduction_factor=1,
        delta=-1.0, n_trials=2, random_state=0, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    for trial in study.trials:
        steps = sorted(trial.intermediate_values)
        assert steps, "each trial must report intermediate OOB scores"
        assert set(steps).issubset(set(ladder))


# --------------------------------------------------------------------------
# Paths not exercised by the contract tests
# --------------------------------------------------------------------------

def test_criterion_is_tuned_when_requested(clf_data):
    """tune_criterion=True must put criterion into the sampled search space."""
    from rf_plateau_hpo.core import tune_rf_oob

    X, y = clf_data
    space = dict(FAST_SPACE)
    space["tune_criterion"] = True

    _, study = tune_rf_oob(
        X, y, "clf", auc_binary, True,
        n_estimators_range=(20, 40), n_trials=3,
        random_state=0, n_jobs=1, verbose=0, refit=False,
        **space,
    )

    sampled = {t.params.get("criterion") for t in study.trials}
    assert sampled <= {"gini", "entropy", "log_loss"}
    assert sampled, "criterion must appear in trial params"


def test_criterion_conflict_is_rejected(clf_data):
    """Passing both tune_criterion=True and an explicit criterion is an error."""
    from rf_plateau_hpo.core import tune_rf_oob

    X, y = clf_data
    space = dict(FAST_SPACE)
    space["tune_criterion"] = True

    with pytest.raises(ValueError, match="criterion"):
        tune_rf_oob(
            X, y, "clf", auc_binary, True,
            criterion="gini",
            n_estimators_range=(20, 40), n_trials=1,
            random_state=0, n_jobs=1, verbose=0, refit=False,
            **space,
        )


def test_class_weight_is_forwarded_to_the_final_model(clf_data):
    from rf_plateau_hpo.core import tune_rf_oob

    X, y = clf_data
    model, study = tune_rf_oob(
        X, y, "clf", auc_binary, True,
        class_weight="balanced",
        n_estimators_range=(20, 40), n_trials=2,
        random_state=0, n_jobs=1, verbose=0,
        **FAST_SPACE,
    )

    if model is None:
        pytest.skip("no completed trials in this run")
    assert model.class_weight == "balanced"


def test_multiclass_problem(clf_data):
    """The wine dataset has three classes; the OOB scorer receives a
    (n_samples, 3) probability matrix rather than two columns."""
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = load_wine(return_X_y=True)
    model, best_n, study, plateau_found = tune_rf_oob_plateau(
        X, y, "clf", auc_multiclass, True,
        n_estimators_start=64, scale_factor=2.0,
        delta=0.9, max_trees=100000,
        n_trials=3, random_state=0, n_jobs=1, verbose=0,
        **FAST_SPACE,
    )

    assert isinstance(plateau_found, bool)
    if plateau_found:
        assert model.n_estimators == best_n
        assert len(model.classes_) == 3


def test_real_dataset_end_to_end():
    """A full run on a real scikit-learn dataset, the way a reader would try it.

    breast_cancer is used rather than iris: it is binary, so the documented
    roc_auc_score(y, proba[:, 1]) scorer applies directly, and it has enough
    samples for out-of-bag estimates to be complete at small forest sizes.
    """
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = load_breast_cancer(return_X_y=True)
    model, best_n, study, plateau_found = tune_rf_oob_plateau(
        X, y, "clf", auc_binary, True,
        n_estimators_start=100, scale_factor=1.5,
        delta=1e-2, max_trees=100000,
        n_trials=6, random_state=42, n_jobs=1, verbose=0,
    )

    if not plateau_found:
        pytest.skip("no plateau within the trial budget for this seed")

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == best_n
    # A tuned forest on this dataset should be clearly better than chance.
    assert study.best_value > 0.95


def test_plateau_search_is_reproducible(clf_data):
    from rf_plateau_hpo.core import tune_rf_oob_plateau

    X, y = clf_data
    kwargs = dict(
        n_estimators_start=64, scale_factor=2.0, delta=1e-2, max_trees=100000,
        n_trials=4, random_state=13, n_jobs=1, verbose=0, refit=False,
        **FAST_SPACE,
    )

    _, n_a, study_a, found_a = tune_rf_oob_plateau(X, y, "clf", auc_binary, True, **kwargs)
    _, n_b, study_b, found_b = tune_rf_oob_plateau(X, y, "clf", auc_binary, True, **kwargs)

    assert (n_a, found_a) == (n_b, found_b)
    assert [t.params for t in study_a.trials] == [t.params for t in study_b.trials]
    assert [t.user_attrs["triplet"] for t in study_a.trials] == \
           [t.user_attrs["triplet"] for t in study_b.trials]


def test_log_file_records_the_best_block(clf_data, tmp_path):
    """The .log files are parsed downstream, so the BEST block must be written."""
    from rf_plateau_hpo.core import tune_rf_oob

    X, y = clf_data
    log_file = tmp_path / "run.log"
    tune_rf_oob(
        X, y, "clf", auc_binary, True,
        n_estimators_range=(20, 40), n_trials=2,
        random_state=0, n_jobs=1, verbose=0, refit=False,
        log_file=str(log_file),
        **FAST_SPACE,
    )

    assert log_file.exists()
    text = log_file.read_text(encoding="utf-8")
    assert "Start tuning" in text
    for key in ("BEST_trial", "BEST_score", "BEST_params"):
        assert key in text
