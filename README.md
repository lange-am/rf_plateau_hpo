# rf-plateau-hpo

Utilities, notebooks, and reproducibility scripts for **Random Forest** hyperparameter optimization with **Optuna**, centered on a triplet-based **PLATEAU** search for a sufficient number of trees (`n_estimators`).

The repository accompanies the paper specified in [`CITATION.cff`](CITATION.cff).

The Python package **`rf_plateau_hpo`** contains:

- `rf_plateau_hpo.core` — the public tuning routines:
  - `tune_rf_oob`: classic Optuna/TPE tuning with `n_estimators` sampled from a fixed range;
  - `tune_rf_oob_bohb`: BOHB-like / Hyperband-style multi-fidelity baseline using `n_estimators` as the resource;
  - `tune_rf_oob_plateau`: the triplet-based PLATEAU search, where TPE samples the non-budget Random Forest hyperparameters and the tree count is adapted internally.
- `rf_plateau_hpo.datasets` — a declarative dataset registry (`data/datasets.yml`) and a local-first dataset loader.
- `notebooks/` — experiment orchestration, analysis helpers, and the full paper-reproducibility notebook.

---

## Installation

```bash
# from the repository root
pip install -e ".[dev]"
```

The minimal runtime package depends on NumPy, pandas, scikit-learn, Optuna, and PyYAML. The `dev` extra additionally installs notebook and experiment-analysis dependencies such as `dill`, `matplotlib`, `seaborn`, `scipy`, `tqdm`, `ucimlrepo`, and Kaggle support.

Python 3.8+ is supported.

---

## Quickstart: loading data as `(X, y)`

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

## RF tuning examples

These examples reflect the public API in `src/rf_plateau_hpo/core.py`.

### 1) Classic OOB/TPE tuning — `tune_rf_oob`

```python
from rf_plateau_hpo.core import tune_rf_oob
from sklearn.metrics import roc_auc_score

auc_binary = lambda y_true, proba: roc_auc_score(y_true, proba[:, 1])

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

For regression, pass a scorer such as `mean_squared_error` and set `greater_is_better=False`.

### 2) Hyperband-style baseline — `tune_rf_oob_bohb`

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

### 3) Triplet-based PLATEAU search — `tune_rf_oob_plateau`

```python
from rf_plateau_hpo.core import tune_rf_oob_plateau

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

The PLATEAU routine does **not** sample `n_estimators` directly in the TPE search space. Instead, TPE samples the remaining Random Forest hyperparameters and the tree count is moved across trials by the internal triplet rule.

---

## Notebooks and experiment scripts

The main reproducibility notebook is [`notebooks/paper_repro.ipynb`](notebooks/paper_repro.ipynb). It contains the end-to-end workflow used for the paper:

- loading and preprocessing all benchmark datasets from `data/datasets.yml`;
- launching experiments through `notebooks/run_experiments.py`;
- comparing TPE, HB, ES, PLATEAU, and decoupled variants;
- generating statistical tables for `n_trials`, `tune_criterion`, `only_depth`, joint-vs-decoupled tuning, pruning, runtime, tree-count cost, and scale-factor sensitivity;
- generating the paper figures, including PLATEAU trajectories, tolerance boxplots, and runtime/tree-count bar plots.

The helper modules in `notebooks/` are ordinary Python files:

- `run_experiments.py` — experiment configuration generation, single-run execution, parsing of study/log metadata, and dataset-level queue execution;
- `analyze_experiments.py` — aggregation of `.dill` files, statistical tests, table export, and plotting utilities;
- `cpu_pinning.py` — Linux-oriented process scheduler with CPU affinity and optional `n_jobs` injection;
- `file_mover.py` — background mover for completed `.dill`/log files from temporary to persistent storage;
- `merge_safe.py`, `split_common_params.py` — small utilities for safe parameter handling.

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

If you use this repository in academic work, please cite the accompanying paper using the metadata in [`CITATION.cff`](CITATION.cff).

---

## License

The source code in this repository is distributed under the MIT License. See [`LICENSE`](LICENSE).

The accompanying paper should be used under the terms of the license specified by the paper venue.
