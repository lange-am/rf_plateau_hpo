# rf-plateau-hpo

Utilities and experiments for **Random Forest** hyperparameter tuning with **Optuna**, including a **plateau search** to find a **sufficient number of trees** (`n_estimators`).

The Python package **`rf_plateau_hpo`** contains:
- `rf_plateau_hpo.core` — the main tuning routines: `tune_rf_oob`, `tune_rf_oob_plateau`.
- `rf_plateau_hpo.datasets` — a declarative **dataset registry** (`data/datasets.yml`) and a loader with local caching/downloading.

---

## Installation

```bash
# from repo root
pip install -e ".[dev]"
# or minimal runtime deps:
# pip install scikit-learn optuna pyyaml pandas
```

Python 3.8+ is supported.

---

## Quickstart: loading data as (X, y)

```python
from pathlib import Path
from rf_plateau_hpo.datasets.dataloader import load_dataset

# Case A: run from repository root
datasets_file = Path("data/datasets.yml").resolve()

# Case B: run from notebooks/ (repo_root/notebooks)
# from pathlib import Path
# ROOT = Path.cwd().parent
# datasets_file = (ROOT / "data" / "datasets.yml").resolve()

# Get (X, y) for a dataset key
X, y = load_dataset("breast_cancer", yml=datasets_file, return_X_y=True)
X.shape, y.shape
```

---

## RF tuning examples (real API)

These examples reflect `src/rf_plateau_hpo/core.py`.

### 1) OOB Optuna tuning — `tune_rf_oob`

```python
from rf_plateau_hpo.core import tune_rf_oob
from sklearn.metrics import roc_auc_score, mean_squared_error

# Classification (AUC, higher is better)
model_clf, study_clf = tune_rf_oob(
    X, y,
    problem="clf",
    score_func=lambda y, proba: roc_auc_score(y, proba[:, 1]),
    greater_is_better=True,
    n_trials=20,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
print("Best value:", study_clf.best_value)
print("Best params:", study_clf.best_params)

# Regression (MSE, lower is better)
# model_reg, study_reg = tune_rf_oob(
#     Xr, yr,
#     problem="reg",
#     score_func=mean_squared_error,
#     greater_is_better=False,
#     n_trials=20,
#     random_state=123,
#     n_jobs=-1,
#     verbose=1,
# )
# print("Best value (MSE):", study_reg.best_value)
# print("Best params:", study_reg.best_params)
```

### 2) Plateau search for `n_estimators` — `tune_rf_oob_plateau`

```python
from rf_plateau_hpo.core import tune_rf_oob_plateau
from sklearn.metrics import roc_auc_score

model_p, best_n_estimators, study_p, plateau_found = tune_rf_oob_plateau(
    X, y,
    problem="clf",
    score_func=lambda y, proba: roc_auc_score(y, proba[:, 1]),
    greater_is_better=True,
    n_trials=15,       # Optuna trials per inner search
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Plateau found:", plateau_found)
print("Best n_estimators:", best_n_estimators)
print("Best value:", study_p.best_value)
print("Best params:", study_p.best_params)
```

> Both functions return a fitted model and an Optuna `study`. The plateau variant additionally returns `(best_n_estimators, plateau_found)`.

---

## Notebooks

See `notebooks/01_quickstart.ipynb` and `notebooks/paper_repro.ipynb`.  
They include:
- Synthetic classification & regression examples with `tune_rf_oob`
- sklearn `breast_cancer` example
- Titanic example using the Kaggle competition (`kaggle-comp:`). If you want a public
  mirror as a fallback, add a separate YAML entry with a `url:` loader and choose it
  explicitly if Kaggle is not configured.

---

## Datasets registry (`data/datasets.yml`)

**Minimal fields per dataset key**

* `name`: human-readable title
* `loader`: how to obtain data (prefixes below)
* `target`: target column name (**used only when `return_X_y=True`; ignored otherwise**)
* `ignored_columns` (optional): columns to drop after reading (comma-separated or YAML list)
* `bib`: BibTeX block for citation

**Local-first behavior & cache layout**

* For non-`file` loaders (`sklearn:`, `uci:`, `kaggle-comp:`, `url:`/`http(s)`), the loader
  looks under `cache/<key>/`. If a file is found, it is loaded and
  **no network request** is made. Otherwise the dataset is fetched and persisted into
  `cache/<key>/` (usually as `<key>.csv`), then loaded from disk.
* For `file` (or `raw`), the loader looks under `raw/<key>/`. We do not auto-read `raw/` for other loader types.
* Both `raw/` and `cache/` live next to the YAML (i.e. `<yaml_dir>/raw`, `<yaml_dir>/cache`).

**UCI column names (important)**

* For `uci:<id>`, on first fetch the loader **attempts** to rename **all columns** to their
  `variables.description` values from UCI metadata.
* Descriptions may be missing for some features; in that case we keep the original names for those features while checking uniqueness.
* The rename is applied **only if** the final set of names is **unique**.
  Otherwise the **original** column names are kept.
* Your YAML `target` **must match the actual column name in the saved file**. For example, for `uci:350` (Default of Credit Card Clients), the original variable name is `"Y"`, and the description-based name is `"default payment next month"`. Typically the loader will save the description-based name, so set `target: "default payment next month"`.

**Supported `loader` prefixes**

* `sklearn:sklearn.datasets.load_breast_cancer`
* `uci:<id>`
* `url:<http(s)://...>` or plain `http(s)://...`
* `kaggle-comp:<slug>@<filename>` (Kaggle **Competitions**; `@filename` is optional. If omitted, the loader auto-picks a good candidate — prefers `train.*` among supported formats; rules must be accepted in the Kaggle UI.)
* `file` (`raw`)

**Supported on-disk formats:** CSV, TXT, TSV/TAB, `.data` (CSV-like), Parquet, JSON, ARFF, XLS/XLSX.  

**Example (excerpt)**

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
    ignored_columns: "PassengerId,Name,Ticket,Cabin"
```

**Ignored columns**  
You can drop columns right after reading from disk by specifying a comma-separated list
(or a YAML list). This does not affect fetching/caching — it only modifies the in-memory DataFrame.
If the target is accidentally included into `ignored_columns` and you call
`load_dataset(..., return_X_y=True)`, an error will be raised because the target column
will be missing — this is expected.

**Local edits & re-fetching**

* Once a dataset is saved on disk (under `<yaml_dir>/cache/<key>/`), you may edit the CSV header
  to rename columns manually.
* The `target` specified in `datasets.yml` **must** match a column name in the saved file;
  otherwise `load_dataset(..., return_X_y=True)` will raise an error.
* While a cached file exists, `load_dataset()` will **not** access the network.
* To force a re-download, delete `<yaml_dir>/cache/<key>/`.

## Kaggle setup (for `kaggle-comp:`)

1. **Create API token:** Kaggle -> **Account -> Create New API Token** -> download `kaggle.json`  
2. **Place the token:**
   - Linux/macOS: `~/.kaggle/kaggle.json` (`chmod 700 ~/.kaggle`, `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   Or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. **Competitions only:** open the competition page and **Accept Rules**.
4. Then you can use `kaggle-comp:` loaders in YAML, and the repository will auto‑download on first use. 
   We use the Kaggle Python API (no CLI required).

---

## License & Citation

- License: MIT (`LICENSE`)
- Citation: see `CITATION.cff`
