"""Tests for the helper modules in notebooks/.

These modules are experiment-orchestration code rather than part of the
installed package, so they are imported by path and the whole file is skipped
when the optional dependencies from the "dev" extra are absent. CI runs it in
the job that installs those.

The emphasis is on the pure functions that shape every experiment directory,
ladder and parsed result: a silent change there would corrupt the analysis
tables without raising anything.
"""

import os
import sys
import time
from pathlib import Path

import pytest

# notebooks/ is a plain directory of scripts, not a package, and its modules
# import each other by bare name.
NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
if not NOTEBOOKS.is_dir():
    pytest.skip("notebooks/ directory not found", allow_module_level=True)
sys.path.insert(0, str(NOTEBOOKS))

pytest.importorskip("dill", reason="notebook helpers need the 'dev' extra")
pytest.importorskip("matplotlib", reason="notebook helpers need the 'dev' extra")
pytest.importorskip("seaborn", reason="notebook helpers need the 'dev' extra")
pytest.importorskip("scipy", reason="notebook helpers need the 'dev' extra")

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import optuna  # noqa: E402


# --------------------------------------------------------------------------
# merge_safe
# --------------------------------------------------------------------------

def test_merge_safe_merges_disjoint_keys():
    from merge_safe import merge_safe

    out = merge_safe({"a": 1}, {"b": 2})
    assert out == {"a": 1, "b": 2}


def test_merge_safe_raises_on_conflicting_values():
    from merge_safe import merge_safe

    with pytest.raises(KeyError, match="key conflict"):
        merge_safe({"a": 1}, {"a": 2})


def test_merge_safe_allows_identical_values():
    from merge_safe import merge_safe

    assert merge_safe({"a": 1}, {"a": 1}) == {"a": 1}


def test_merge_safe_conflict_modes():
    from merge_safe import merge_safe

    # "ok" overwrites silently
    assert merge_safe({"a": 1}, {"a": 2}, on_conflict="ok") == {"a": 2}

    with pytest.warns(RuntimeWarning, match="key conflict"):
        merge_safe({"a": 1}, {"a": 2}, on_conflict="warn")

    with pytest.raises(ValueError, match="on_conflict"):
        merge_safe({"a": 1}, {"b": 2}, on_conflict="nonsense")


def test_merge_safe_uses_identity_for_arrays():
    """Two equal-but-distinct arrays must count as a conflict, not a match.

    Comparing them by value would raise on ambiguous truth; comparing by
    identity is what lets X and y be passed as shared parameters safely.
    """
    from merge_safe import merge_safe

    a = np.zeros(3)
    b = np.zeros(3)
    with pytest.raises(KeyError):
        merge_safe({"X": a}, {"X": b})
    assert merge_safe({"X": a}, {"X": a}) == {"X": a}


# --------------------------------------------------------------------------
# split_common_params
# --------------------------------------------------------------------------

def test_split_common_params_extracts_shared_scalars():
    from split_common_params import split_common_params

    configs = [
        {"method": "TPE", "n_trials": 40, "seed": 1},
        {"method": "PLATEAU", "n_trials": 40, "seed": 2},
    ]
    common, per_run = split_common_params(configs)

    assert common == {"n_trials": 40}
    assert per_run == [
        {"method": "TPE", "seed": 1},
        {"method": "PLATEAU", "seed": 2},
    ]


def test_split_common_params_shares_arrays_by_identity():
    from split_common_params import split_common_params

    X = np.zeros(3)
    common, per_run = split_common_params([{"X": X, "i": 1}, {"X": X, "i": 2}])
    assert "X" in common and common["X"] is X
    assert per_run == [{"i": 1}, {"i": 2}]

    # A distinct but equal array is not shared.
    common2, _ = split_common_params([{"X": np.zeros(3)}, {"X": np.zeros(3)}])
    assert common2 == {}


def test_split_common_params_on_empty_input():
    from split_common_params import split_common_params

    assert split_common_params([]) == ({}, [])


# --------------------------------------------------------------------------
# build_ladder / get_experiment_directory
# --------------------------------------------------------------------------

def test_build_ladder_matches_the_documented_example():
    """run_experiments.py documents this exact ladder for sf=1.5, T0=100."""
    from run_experiments import build_ladder

    ladder = build_ladder(1.5, 100, 2000)
    assert ladder == (100, 150, 225, 338, 507, 760, 1140, 1710, 2565)


def test_build_ladder_is_strictly_increasing_and_covers_t_max():
    from run_experiments import build_ladder

    for sf in (1.25, 1.5, 1.75, 2.0):
        ladder = build_ladder(sf, 100, 2000)
        assert list(ladder) == sorted(set(ladder)), f"sf={sf} not strictly increasing"
        assert ladder[0] == 100
        assert ladder[-1] >= 2000
        assert ladder[-2] < 2000


def test_experiment_directory_encodes_the_configuration():
    from run_experiments import get_experiment_directory

    p = get_experiment_directory(
        dataset="mydata", tune_criterion=True, depth_trees_only=False,
        method="PLATEAU", scale_factor=1.5, delta=1e-3, n_trials=120,
    )
    parts = p.parts
    assert parts[0] == "mydata"
    assert "tune_criterion=True" in parts
    assert "depth_trees_only=False" in parts
    assert "PLATEAU" in parts
    assert "scale_factor=1.5" in parts
    assert "delta=1e-3" in parts
    assert parts[-1] == "n_trials=120"


def test_experiment_directory_omits_delta_for_methods_without_it():
    from run_experiments import get_experiment_directory

    p = get_experiment_directory(
        dataset="mydata", tune_criterion=True, depth_trees_only=False,
        method="TPE", scale_factor=1.5, delta=1e-3, n_trials=120,
    )
    assert not any(part.startswith("delta=") for part in p.parts)


# --------------------------------------------------------------------------
# parse_log_tail
# --------------------------------------------------------------------------

def test_parse_log_tail_reads_the_final_best_block(tmp_path):
    from run_experiments import parse_log_tail

    log = tmp_path / "run.log"
    log.write_text(
        "2026-01-01 00:00:00 | INFO | Start tuning | method=PLATEAU [t+0.001s]\n"
        "2026-01-01 00:00:01 | INFO | trees=  32 | oob_score=0.900000 [t+1.000s]\n"
        "2026-01-01 00:00:02 | INFO | BEST_trial=3 [t+2.000s]\n"
        "2026-01-01 00:00:02 | INFO | BEST_score=0.987654 [t+2.100s]\n"
        "2026-01-01 00:00:02 | INFO | BEST_n_estimators=64 [t+2.200s]\n"
        "2026-01-01 00:00:02 | INFO | BEST_triplet=(32, 64, 128) [t+2.345s]\n",
        encoding="utf-8",
    )

    out = parse_log_tail(log)
    assert out["BEST_trial"] == 3
    assert out["BEST_score"] == pytest.approx(0.987654)
    assert out["BEST_n_estimators"] == 64
    assert out["BEST_triplet"] == (32, 64, 128)
    assert out["time_total"] == pytest.approx(2.345)


def test_parse_log_tail_on_a_log_without_a_best_block(tmp_path):
    from run_experiments import parse_log_tail

    log = tmp_path / "empty.log"
    log.write_text("2026-01-01 00:00:00 | INFO | Start tuning [t+0.001s]\n", encoding="utf-8")
    assert parse_log_tail(log) == {}


# --------------------------------------------------------------------------
# parse_study
# --------------------------------------------------------------------------

def _study_with_attrs(records):
    """Build a real Optuna study whose trials carry the given user_attrs."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    it = iter(records)

    def objective(trial):
        rec = next(it)
        for k, v in rec.items():
            trial.set_user_attr(k, v)
        if rec.get("_prune"):
            raise optuna.TrialPruned()
        return rec.get("_value", 0.5)

    study.optimize(objective, n_trials=len(records))
    return study


def test_parse_study_counts_shifts_and_pruning():
    from run_experiments import parse_study

    study = _study_with_attrs([
        {"trees_built": 100, "triplet": (50, 100, 200), "shift": -1, "_value": 0.8},
        {"trees_built": 200, "triplet": (100, 200, 400), "shift": 0, "_value": 0.9},
        {"trees_built": 400, "triplet": (200, 400, 800), "shift": 1,
         "pruned": "no_plateau", "_prune": True},
    ])

    out = parse_study(study)
    assert out["n_trials_total"] == 3
    assert out["n_trees_built"] == 700
    assert out["n_trials_shift_left"] == 1
    assert out["n_trials_stay"] == 1
    assert out["n_trials_shift_right"] == 1
    assert out["n_trials_pruned"] == 1
    assert out["n_trials_pruned_no_plateau"] == 1
    assert out["B"] == [100, 200, 400]
    assert out["pruned"] == [False, False, True]
    assert out["BEST_score"] == pytest.approx(0.9)


def test_parse_study_warns_when_trees_built_is_missing():
    from run_experiments import parse_study

    study = _study_with_attrs([{"_value": 0.5}])
    with pytest.warns(RuntimeWarning, match="trees_built missing"):
        parse_study(study)


def test_sumup_common_adds_counters_only():
    from run_experiments import sumup_common

    d1 = {"n_trees_built": 100, "time_total": 1.5, "BEST_score": 0.9}
    d2 = {"n_trees_built": 200, "time_total": 2.5, "BEST_score": 0.95}
    out = sumup_common(d1, d2)

    assert out["n_trees_built"] == 300
    assert out["time_total"] == pytest.approx(4.0)
    assert "BEST_score" not in out, "non-counter keys must not be summed"


# --------------------------------------------------------------------------
# file_mover
# --------------------------------------------------------------------------

def test_file_mover_moves_completed_files_and_skips_temporaries(tmp_path):
    from file_mover import FileMoverThread

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "done.dill").write_bytes(b"x")
    (src / "sub" / "running.dill.tmp").write_bytes(b"y")

    mover = FileMoverThread(
        source_dir=src, dest_dir=dst, min_age=0, check_interval=0.05,
        ignore_ext=".tmp", total_files=1, autostart=True,
    )
    try:
        mover.wait_for_completion(target=1, timeout=10)
    finally:
        mover.stop(timeout=5)

    assert (dst / "sub" / "done.dill").exists(), "completed file must be moved"
    assert (src / "sub" / "running.dill.tmp").exists(), "temporary file must stay"
    assert mover.moved_count == 1


def test_file_mover_requires_a_target():
    from file_mover import FileMoverThread

    mover = FileMoverThread(source_dir=".", dest_dir=".", autostart=False)
    with pytest.raises(ValueError, match="No target specified"):
        mover.wait_for_completion()


# --------------------------------------------------------------------------
# run_experiment: the orchestration layer end to end
# --------------------------------------------------------------------------

ALL_METHODS = [
    "TPE", "PLATEAU", "HB", "ES",
    "TPE_Tmin", "TPE_Tmin-Tmax", "TPE_Tmin-ES", "TPE_Tmin-PLT",
]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_run_experiment_produces_log_and_dill(method, tmp_path):
    """Every protocol in the paper must run and leave parseable artefacts.

    A .log.tmp / .dill.tmp pair is renamed to .log / .dill only on success, so
    the presence of the final names is itself the completion signal the file
    mover relies on.
    """
    import warnings

    import dill
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from run_experiments import run_experiment

    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=4, random_state=0
    )
    auc = lambda y_true, proba: roc_auc_score(y_true, proba[:, 1])  # noqa: E731

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_experiment(
            method, X, y, "clf", auc, True,
            n_estimators_range=(20, 80),
            n_estimators_ladder=(20, 40, 80),
            n_estimators_start=20, scale_factor=2.0, delta=1e-2, max_trees=2000,
            hyperband_reduction_factor=1 if method in ("ES", "TPE_Tmin-ES") else 3,
            max_features_grid=("sqrt",), max_depth_range=(3, 8),
            min_samples_leaf_range=(1, 5), min_samples_split_range=(2, 10),
            tune_criterion=False, n_trials=3, random_state=0, n_jobs=1,
            verbose=0, outdir=tmp_path, dataset="synth",
        )

    dills = list(tmp_path.glob("*.dill"))
    logs = list(tmp_path.glob("*.log"))
    assert len(dills) == 1, f"expected one .dill, got {[p.name for p in dills]}"
    # Two-stage methods write one log per stage.
    assert len(logs) == len(method.split("-"))
    assert not list(tmp_path.glob("*.tmp")), "temporary files must be renamed"

    with open(dills[0], "rb") as f:
        payload = dill.load(f)

    assert payload["method"] == method
    assert payload["params_data"]["dataset"] == "synth"
    assert "X" not in payload["params_in"], "raw data must not be serialised"
    assert "y" not in payload["params_in"]
    assert "study" in payload["params_out"]


def test_run_experiment_rejects_an_unknown_method(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from run_experiments import run_experiment

    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    auc = lambda y_true, proba: roc_auc_score(y_true, proba[:, 1])  # noqa: E731

    with pytest.raises(ValueError, match="Unknown method"):
        run_experiment(
            "NOT_A_METHOD", X, y, "clf", auc, True,
            n_trials=1, random_state=0, n_jobs=1, verbose=0,
            outdir=tmp_path, dataset="synth",
        )


# --------------------------------------------------------------------------
# cpu_pinning
# --------------------------------------------------------------------------

def test_cpu_allocation_produces_disjoint_blocks():
    from cpu_pinning import _allocate_run_cpu_blocks, _get_allowed_cpus

    cpus = _get_allowed_cpus()
    blocks, info = _allocate_run_cpu_blocks(cpus, 1, "auto", "prefer")

    assert blocks, "at least one run slot must be allocated"
    flat = [c for b in blocks for c in b]
    assert len(flat) == len(set(flat)), "run slots must not share logical CPUs"
    assert set(flat).issubset(set(cpus))
    assert info["n_run_slots"] == len(blocks)


def test_cpu_allocation_rejects_invalid_arguments():
    from cpu_pinning import _allocate_run_cpu_blocks, _get_allowed_cpus

    cpus = _get_allowed_cpus()
    with pytest.raises(ValueError, match="n_phys_cores_per_run"):
        _allocate_run_cpu_blocks(cpus, 0, "auto", "prefer")
    with pytest.raises(ValueError, match="socket_policy"):
        _allocate_run_cpu_blocks(cpus, 1, "auto", "nonsense")


def test_run_queue_pinned_refuses_cpus_outside_the_affinity_mask():
    """Guards against bypassing a Slurm or cgroup restriction."""
    from cpu_pinning import run_queue_pinned, _get_allowed_cpus

    outside = max(_get_allowed_cpus()) + 1000
    with pytest.raises(ValueError, match="outside the current process affinity"):
        run_queue_pinned(lambda **kw: None, [{}], allowed_cpus=[outside])


def test_run_queue_pinned_executes_every_task(tmp_path):
    from cpu_pinning import run_queue_pinned

    results = run_queue_pinned(
        _square, [{"i": i} for i in range(6)],
        n_phys_cores_per_run=1, static_params={"offset": 10},
        return_outputs=True, verbose=0,
    )
    assert results == [i * i + 10 for i in range(6)]


def _square(i, offset=0, n_jobs=None):
    """Module-level so it stays picklable under the spawn start method."""
    return i * i + offset
