# How Many Trees in a Random Forest? Adaptive `n_estimators` Tuning with PLATEAU Search and Optuna

[![tests](https://github.com/lange-am/rf_plateau_hpo/actions/workflows/ci.yml/badge.svg)](https://github.com/lange-am/rf_plateau_hpo/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rf-plateau-hpo.svg)](https://pypi.org/project/rf-plateau-hpo/)
[![Conda-forge](https://img.shields.io/conda/vn/conda-forge/rf-plateau-hpo.svg)](https://anaconda.org/conda-forge/rf-plateau-hpo)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/alange/how-many-trees-does-a-random-forest-need)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![arXiv (method)](https://img.shields.io/badge/arXiv-2606.03549-b31b1b.svg)](https://arxiv.org/abs/2606.03549)
[![arXiv (theory)](https://img.shields.io/badge/arXiv-2606.30837-b31b1b.svg)](https://arxiv.org/abs/2606.30837)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2026.3705574-blue.svg)](https://doi.org/10.1109/ACCESS.2026.3705574)
[![IEEE Access](https://img.shields.io/badge/IEEE%20Access-Open%20Access-00629B.svg)](https://ieeexplore.ieee.org/document/11571780)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.22847391.svg)](https://doi.org/10.5281/zenodo.22847391)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/lange-am/rf_plateau_hpo/)](https://archive.softwareheritage.org/browse/origin/https://github.com/lange-am/rf_plateau_hpo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🚀 **[Quickstart notebook](notebooks/quickstart.ipynb)** · [Kaggle](https://www.kaggle.com/code/alange/how-many-trees-does-a-random-forest-need) · [Conda-forge](https://anaconda.org/conda-forge/rf-plateau-hpo) · 📄 [Project page](https://lange-am.github.io/rf_plateau_hpo/) · [Paper](https://doi.org/10.1109/ACCESS.2026.3705574) · [Preprint](https://arxiv.org/abs/2606.03549)

**How many trees should a Random Forest use?** Stop guessing a range for
`n_estimators`. **PLATEAU search** moves along the tree-count axis until
substantially more trees stop buying meaningful OOB score, while **Optuna/TPE**
tunes the remaining Random Forest hyperparameters. No sampled `[T_min, T_max]`
range.

It is the reference implementation for the
[IEEE Access 2026 paper](https://doi.org/10.1109/ACCESS.2026.3705574)
([arXiv:2606.03549](https://arxiv.org/abs/2606.03549)).

```python
from rf_plateau_hpo.core import tune_rf_oob_plateau

# Assuming X, y and auc_binary are defined.
# No [T_min, T_max] range for n_estimators — the tree count is found adaptively.
model, best_n_estimators, study, plateau_found = tune_rf_oob_plateau(
    X, y, problem="clf", score_func=auc_binary, greater_is_better=True,
    n_estimators_start=100, scale_factor=1.5, delta=1e-3, n_trials=20,
)
print(best_n_estimators)   # near-minimal sufficient number of trees
```

---

## Why tuning `n_estimators` over a fixed range is the wrong tool

A Random Forest does not give HPO a useful interior optimum in `n_estimators`.
The score rises, then plateaus. Once you are on that plateau, a fixed-range
optimizer is mostly choosing among nearly indistinguishable forests, so the
answer becomes tied to the arbitrary `T_max` you typed in.

Set `T_max` too low and the forest may be insufficient. Set it too high and
you pay for trees that no longer move the score.

**PLATEAU does not search for the best tree count inside a range. It searches
for the plateau itself.** It evaluates out-of-bag (OOB) scores at geometrically
spaced tree counts and moves the triplet using their **relative score changes**
and a **relative tolerance** ε (`delta` in the API).

Instead of asking “which tree count wins inside my guessed range?”, PLATEAU asks
“when are more trees no longer worth it?”

---

## Papers

### 1. PLATEAU search, Optuna integration, and experiments

V. A. Porvatov, A. A. Dukhovny, and A. M. Lange,
**“How Many Trees in a Random Forest? A Revisited Approach With Plateau Search
and Optuna Integration,”**
*IEEE Access*, vol. 14, pp. 93670–93693, 2026.

- DOI: [10.1109/ACCESS.2026.3705574](https://doi.org/10.1109/ACCESS.2026.3705574)
- IEEE Xplore: [document 11571780](https://ieeexplore.ieee.org/document/11571780)
- Preprint (free full text): [arXiv:2606.03549](https://arxiv.org/abs/2606.03549)
  · [PDF](https://arxiv.org/pdf/2606.03549) · [v1](https://arxiv.org/abs/2606.03549v1)

The paper introduces triplet-based PLATEAU search for adaptive `n_estimators`
selection and joint Random Forest hyperparameter tuning with Optuna/TPE, and
benchmarks it against TPE with a fixed range, Hyperband/BOHB-style
multi-fidelity tuning, and OOB early stopping.

### 2. Stationary-distribution theory

A. A. Dukhovny and A. M. Lange,
**“A Stationary-Distribution Theory for Triplet-Based Plateau Search in Random
Forest Ensemble-Size Selection,”** 2026.

- Preprint: [arXiv:2606.30837](https://arxiv.org/abs/2606.30837)
  · [PDF](https://arxiv.org/pdf/2606.30837)

The theory models adaptive plateau search as a stochastic process and
characterizes its stationary distribution, **equilibrium number of trees**, and
dependence on the relative tolerance ε and the geometric scale factor.

Machine-readable citation metadata for the software and the IEEE Access paper
are provided in [`CITATION.cff`](CITATION.cff); BibTeX is at the
[end of this file](#citation).

---

## How PLATEAU search works

For a current ensemble size `B` and scale factor `sf`, PLATEAU evaluates an
approximately geometric triplet

`B / sf`, `B`, `B * sf`.

At each HPO trial, Optuna/TPE samples the non-budget Random Forest
hyperparameters while PLATEAU compares OOB scores across the triplet. The
relative score changes determine whether the tree-count search shifts left,
stays, or shifts right.

The triplet therefore adapts across trials and concentrates around an ensemble
size where substantially more trees yield only tolerance-level score changes.

`delta` is the relative tolerance ε used by the plateau criterion, and
`scale_factor` controls the spacing of candidate ensemble sizes. Unlike
standard Optuna tuning of `n_estimators`, PLATEAU does not require a sampled
hyperparameter range `[T_min, T_max]` for the number of trees. `max_trees` is
only a safety bound on feasible ensemble sizes.

---

## Frequently asked questions

### How many trees does a Random Forest need?

There is no universal number. The answer depends on the dataset, the scoring
metric, the other hyperparameters (tree depth, `max_features`), and on how much
score change you are willing to ignore. PLATEAU search makes that last quantity
explicit: you specify a relative tolerance ε, and the algorithm seeks a
near-minimal sufficient ensemble size at that tolerance.

### Is the scikit-learn default of `n_estimators=100` enough?

Sometimes, but it is a default rather than a decision. Whether 100 trees is
sufficient is exactly what the plateau criterion tests, and the sufficient size
also depends on the *other* hyperparameters being tuned at the same time —
which is why PLATEAU runs inside the HPO loop rather than before it.

### Can too many trees make a Random Forest overfit?

Adding trees mainly reduces variance and does not drive the usual
overfitting-with-capacity behaviour, so the score rises and then flattens.
**More trees are usually not the problem. Paying for trees that no longer move
the score is.** The bill shows up in training time, memory, and inference latency.

### What range should I give Optuna for `n_estimators`?

With PLATEAU search, none — that is the point. `n_estimators` is removed from
the TPE search space entirely; you give a starting size, a scale factor, a
tolerance, and a safety bound (`max_trees`).

### How is this different from OOB early stopping?

Early-stopping strategies also avoid fixing a range, but they read a single
score curve and can be sensitive to score noise and prone to premature
stopping. PLATEAU instead accumulates information *across HPO trials* and uses
a triplet of forest sizes, so the tree count and the other hyperparameters
adapt to each other.

### Does it work for regression?

Yes. Pass `problem="reg"` and a scorer such as `mean_squared_error` with
`greater_is_better=False`.

### I use R (`randomForest`, `ranger`). Does this apply?

The idea does: `ntree` / `num.trees` is the same quantity as `n_estimators`,
and the plateau criterion only needs an OOB score at a few forest sizes. This
repository provides the Python/scikit-learn/Optuna implementation only.

### What value of `delta` (ε) should I pick?

It is the relative score change you are willing to treat as negligible; `1e-3`
is a reasonable starting point for AUC-type metrics. The tolerance directly
controls the resulting ensemble size — the stationary-distribution paper
([arXiv:2606.30837](https://arxiv.org/abs/2606.30837)) characterizes that
dependence, and the sensitivity experiments in the IEEE Access paper measure it
empirically.

---

## Related work

Research on the number of trees in a Random Forest goes back to Breiman (2001)
and includes Oshiro, Perez and Baranauskas (2012), *How Many Trees in a Random
Forest?*, Latinne et al. (2001), Hernández-Lobato et al. (2013), and the
tuning survey of Probst, Wright and Boulesteix (2019). These works study forest size and related Random Forest tuning questions,
typically separately from joint adaptive ensemble-size selection within HPO.

The contribution here is to treat the ensemble size and the other Random Forest
hyperparameters as interdependent, and to resolve both inside a single
Optuna/TPE run — without a predefined `[T_min, T_max]` range for the number of
trees. See Section II of the
[paper](https://arxiv.org/abs/2606.03549) for the full comparison.

---

## Package overview

The Python package **`rf_plateau_hpo`** contains:

- `rf_plateau_hpo.core` — the public Random Forest tuning routines:
  - `tune_rf_oob_plateau` — adaptive triplet-based PLATEAU search; TPE tunes
    the non-budget hyperparameters while `n_estimators` is adapted internally;
  - `tune_rf_oob` — classic Optuna/TPE tuning with `n_estimators` sampled from
    a fixed range;
  - `tune_rf_oob_bohb` — BOHB-like / Hyperband-style multi-fidelity baseline
    using `n_estimators` as the resource.
- `rf_plateau_hpo.datasets` — a declarative dataset registry
  (`data/datasets.yml`) and a local-first dataset loader.
- `notebooks/` — the two-minute quickstart, experiment orchestration and
  analysis helpers, and the full paper-reproducibility notebook.

---

## Installation

For regular use, install the package from PyPI:

```bash
pip install rf-plateau-hpo
```

or from conda-forge:

```bash
conda install -c conda-forge rf-plateau-hpo
```

For development or to run the full notebook/reproducibility stack from a
repository clone, install the `dev` extra instead:

```bash
pip install -e ".[dev]"
```

The minimal runtime package depends on NumPy, pandas, scikit-learn, Optuna,
and PyYAML. The `dev` extra additionally installs notebook and
experiment-analysis dependencies such as `dill`, `matplotlib`, `seaborn`,
`scipy`, `tqdm`, `ucimlrepo`, and Kaggle support.

Python 3.8+ is supported.

---

## Loading repository datasets as `(X, y)`

```python
from pathlib import Path
from rf_plateau_hpo.datasets.dataloader import load_dataset

# Case A: run from the repository root
datasets_file = Path("data/datasets.yml").resolve()

# Case B: run from notebooks/ (repo_root/notebooks)
# ROOT = Path.cwd().parent
# datasets_file = (ROOT / "data" / "datasets.yml").resolve()

X, y = load_dataset("breast_cancer", yml=datasets_file, return_X_y=True)
print(X.shape, y.shape)
```

---

## Random Forest tuning examples

These examples reflect the public API in `src/rf_plateau_hpo/core.py`.

### 1) Adaptive `n_estimators` tuning — `tune_rf_oob_plateau`

PLATEAU removes `n_estimators` from the direct TPE search space. Optuna tunes
the remaining Random Forest hyperparameters while the ensemble size is adapted
by the triplet search.

```python
from rf_plateau_hpo.core import tune_rf_oob_plateau
from sklearn.metrics import roc_auc_score

auc_binary = lambda y_true, proba: roc_auc_score(y_true, proba[:, 1])

model_p, best_n_p, study_p, plateau_found = tune_rf_oob_plateau(
    X,
    y,
    problem="clf",
    score_func=auc_binary,
    greater_is_better=True,
    n_estimators_start=100,
    scale_factor=1.5,
    delta=1e-3,
    max_trees=100000,
    n_trials=20,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Plateau found:", plateau_found)
print("Best n_estimators:", best_n_p)
print("Best value:", study_p.best_value)
print("Best params:", study_p.best_params)
```

If `plateau_found` is `False`, no trial reached a plateau within the allowed
budget: increase `n_trials` or `max_trees`, or adjust the starting parameters.

### 2) Fixed-range Optuna/TPE tuning — `tune_rf_oob`

Conventional Random Forest hyperparameter optimization samples `n_estimators`
jointly with the other hyperparameters from a predefined range.

```python
from rf_plateau_hpo.core import tune_rf_oob

model_tpe, study_tpe = tune_rf_oob(
    X,
    y,
    problem="clf",
    score_func=auc_binary,
    greater_is_better=True,
    n_estimators_range=(100, 2565),
    n_trials=20,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Best value:", study_tpe.best_value)
print("Best params:", study_tpe.best_params)
```

For regression, pass a scorer such as `mean_squared_error` and set
`greater_is_better=False`.

### 3) Hyperband-style baseline — `tune_rf_oob_bohb`

This multi-fidelity baseline uses `n_estimators` as the resource and evaluates
a fixed ladder of increasing ensemble sizes.

```python
from rf_plateau_hpo.core import tune_rf_oob_bohb

model_hb, best_n_hb, study_hb, stopped_hb = tune_rf_oob_bohb(
    X,
    y,
    problem="clf",
    score_func=auc_binary,
    greater_is_better=True,
    n_estimators_ladder=(100, 150, 225, 338, 507, 760, 1140, 1710, 2565),
    hyperband_reduction_factor=3,
    n_trials=20,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Best value:", study_hb.best_value)
print("Best n_estimators:", best_n_hb)
```

---

## Notebooks and experiment scripts

Start with **[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb)** — a
self-contained, roughly two-minute tutorial on scikit-learn's Breast Cancer
dataset. Run it
[on Kaggle](https://www.kaggle.com/code/alange/how-many-trees-does-a-random-forest-need)
or
[open it directly in Colab](https://colab.research.google.com/github/lange-am/rf_plateau_hpo/blob/main/notebooks/quickstart.ipynb).

The Kaggle notebook is kept in sync with this repository: changes to
`notebooks/quickstart.ipynb` on `main` are published automatically as a new
Kaggle version.

It:

- plots the OOB curve as it rises and then flattens into noise;
- runs the same fixed-range TPE tuning with two different `T_max` values;
- runs PLATEAU without a sampled `n_estimators` range;
- visualizes the triplet walk across optimization trials;
- compares final forest size and the total number of trees grown while tuning.

**In the reference run used to build the tutorial:** fixed-range TPE on
`[100, 2565]` selects 688 trees at OOB AUC 0.9909; PLATEAU selects 30 at
0.9899 and grows about **12× fewer trees** while tuning. That is one dataset
and one seed — the paper contains the benchmark.

For full reproducibility, see
[`notebooks/paper_repro.ipynb`](notebooks/paper_repro.ipynb). It contains the
end-to-end workflow used for the IEEE Access paper:

- loading and preprocessing all benchmark datasets from `data/datasets.yml`;
- launching experiments through `notebooks/run_experiments.py`;
- comparing TPE, HB, ES, PLATEAU, and decoupled variants;
- generating statistical tables for `n_trials`, `tune_criterion`,
  `only_depth`, joint-vs-decoupled tuning, pruning, runtime, tree-count cost,
  and scale-factor sensitivity;
- generating the paper figures, including PLATEAU trajectories, tolerance
  boxplots, and runtime/tree-count bar plots.

The helper modules in `notebooks/` are ordinary Python files:

- `run_experiments.py` — experiment configuration generation, single-run
  execution, parsing of study/log metadata, and dataset-level queue execution;
- `analyze_experiments.py` — aggregation of `.dill` files, statistical tests,
  table export, and plotting utilities;
- `cpu_pinning.py` — Linux-oriented process scheduler with CPU affinity and
  optional `n_jobs` injection;
- `file_mover.py` — background mover for completed `.dill`/log files from
  temporary to persistent storage;
- `merge_safe.py`, `split_common_params.py` — small utilities for safe
  parameter handling.

---

## Datasets registry (`data/datasets.yml`)

**Minimal fields per dataset key**

- `name`: human-readable title;
- `loader`: how to obtain data;
- `target`: target column name, used only when `return_X_y=True`;
- `ignored_columns` (optional): columns to drop after reading;
- `bib`: BibTeX block for dataset citation.

**Local-first behavior and cache layout**

- For non-local loaders (`sklearn:`, `uci:`, `kaggle-comp:`, `url:` / `http(s)`), the loader first checks `cache/<key>/`. If a supported file is found, it is loaded and no network request is made. Otherwise, the dataset is fetched, saved under `cache/<key>/`, and loaded from disk.
- For `file` or `raw` loaders, the loader checks `raw/<key>/`.
- We do not auto-read `raw/` for other loader types.
- Both `raw/` and `cache/` live next to the YAML file, i.e. under `<yaml_dir>/raw` and `<yaml_dir>/cache`.

**UCI column names**

- For `uci:<id>`, on first fetch the loader attempts to rename all columns to their `variables.description` values from UCI metadata.
- Descriptions may be missing for some features; in that case the original names are kept for those features while checking uniqueness.
- The rename is applied only if the final set of names is unique. Otherwise the original column names are kept.
- The YAML `target` must match the actual column name in the saved file. For example, for `uci:350` (Default of Credit Card Clients), the original variable name is `"Y"`, while the description-based name is `"default payment next month"`. Typically the loader saves the description-based name, so the registry sets `target: "default payment next month"`.

**Supported `loader` prefixes**

- `sklearn:sklearn.datasets.<dataset>`;
- `uci:<id>` via `ucimlrepo`;
- `url:<http(s)://...>` or a plain `http(s)://...` URL;
- `kaggle-comp:<slug>@<filename>` for Kaggle competitions (`@filename` is optional; if omitted, the loader attempts to select a suitable file, preferring `train.*` among supported formats);
- `file` / `raw` for locally stored files.

Supported on-disk formats include CSV, TXT, TSV/TAB, `.data`, Parquet, JSON, ARFF, XLS, and XLSX.

**Example**

```yaml
datasets:
  breast_cancer:
    name: "Breast Cancer Wisconsin (Diagnostic)"
    loader: "sklearn:sklearn.datasets.load_breast_cancer"
    target: "target"  # used only when return_X_y=True

  credit_card_default:
    name: "Default of Credit Card Clients (Taiwan)"
    loader: "uci:350"  # UCI dataset id 350 – https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    target: "default payment next month"  # description-based column name (applied if unique)
    ignored_columns: "ID"

  titanic:
    name: "Titanic (Kaggle competition)"
    loader: "kaggle-comp:titanic@train.csv"
    target: "Survived"
    ignored_columns: "PassengerId,Name,Ticket"
```

**Ignored columns**

You can drop columns right after reading from disk by specifying a comma-separated list or a YAML list. This does not affect fetching/caching; it only modifies the in-memory DataFrame. If the target is accidentally included in `ignored_columns` and you call `load_dataset(..., return_X_y=True)`, an error will be raised because the target column will be missing. This is expected.

**Local edits and re-fetching**

- Once a dataset is saved on disk under `<yaml_dir>/cache/<key>/`, you may edit the CSV header to rename columns manually.
- The `target` specified in `datasets.yml` must match a column name in the saved file; otherwise, `load_dataset(..., return_X_y=True)` will raise an error.
- While a cached file exists, `load_dataset()` will not access the network.
- To force a re-download, delete `<yaml_dir>/cache/<key>/`.

**Kaggle setup**

1. Create an API token in Kaggle: **Account → Create New API Token**.
2. Place `kaggle.json` under `~/.kaggle/kaggle.json` on Linux/macOS or `C:\Users\<you>\.kaggle\kaggle.json` on Windows, or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. For competitions, accept the competition rules in the Kaggle UI.
4. Then `kaggle-comp:` entries in `data/datasets.yml` can be fetched automatically on first use.

---

## Citation

If you use this repository or the method in academic work, please cite the
IEEE Access paper:

```bibtex
@article{porvatov2026howmanytrees,
  author  = {Porvatov, V. A. and Dukhovny, A. A. and Lange, A. M.},
  title   = {How Many Trees in a Random Forest? A Revisited Approach With
             Plateau Search and Optuna Integration},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {93670--93693},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3705574},
  url     = {https://doi.org/10.1109/ACCESS.2026.3705574},
  eprint  = {2606.03549},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

For the stationary-distribution analysis of triplet-based plateau search:

```bibtex
@article{dukhovny2026stationary,
  author  = {Dukhovny, A. A. and Lange, A. M.},
  title   = {A Stationary-Distribution Theory for Triplet-Based Plateau Search
             in Random Forest Ensemble-Size Selection},
  journal = {arXiv preprint arXiv:2606.30837},
  year    = {2026},
  eprint  = {2606.30837},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url     = {https://arxiv.org/abs/2606.30837}
}
```

Machine-readable metadata is also available in [`CITATION.cff`](CITATION.cff);
GitHub renders it under **Cite this repository** in the sidebar.

---

## License

The source code in this repository is distributed under the MIT License. See [`LICENSE`](LICENSE).

The accompanying paper should be used under the terms of the license specified by the paper venue.
