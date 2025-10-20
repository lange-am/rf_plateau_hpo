"""
rf_plateau_hpo.datasets.dataloader
----------------------------------

A small utility module that loads tabular datasets described in a YAML registry,
with *local-first* behavior and automatic on-demand fetching + caching.

What it does
============
- You list datasets in a single YAML file (e.g. `data/datasets.yml`) with minimal
  metadata: a human-readable name, a loader string, an optional BibTeX citation,
  and (when you want `(X, y)`) the name of the target column.
- When you call `load_dataset(key, yml=...)`, the loader:
  1) Looks for files under `cache/<key>/` (next to your YAML) and loads from disk if present.
  2) Otherwise downloads/extracts the dataset according to the `loader` and persists it
     under `cache/<key>/`, then loads from disk.
- For UCI (`uci:<id>`), on first fetch we attempt to rename *all* columns to the
  `variables.description` provided by UCI; this rename is applied only if all resulting
  names are unique (otherwise original column names are kept).
- The `target` from YAML is **used only when `return_X_y=True`**; otherwise it is ignored.

Supported on-disk formats
=========================
CSV, TXT, **TSV/TAB**, **.data** (CSV-like), **Parquet**, **JSON**, **ARFF**, **XLS/XLSX**.

Minimal YAML example
====================
```yaml
datasets:
  breast_cancer:
    name: "Breast Cancer Wisconsin (Diagnostic)"
    loader: "sklearn:sklearn.datasets.load_breast_cancer"
    target: "target"

  credit_card_default:
    name: "Default of Credit Card Clients (Taiwan)"
    loader: "uci:350"
    target: "default payment next month"   # description-based name for uci:350
    ignored_columns: "ID"

  titanic:
    name: "Titanic (Kaggle competition)"
    loader: "kaggle-comp:titanic@train.csv"
    target: "Survived"
    ignored_columns: "PassengerId,Name,Ticket,Cabin"
```

Basic usage
===========
```python
from pathlib import Path
from rf_plateau_hpo.datasets.dataloader import load_dataset

yaml_file = Path("data/datasets.yml")

# Load (X, y) for a dataset key
X, y = load_dataset("breast_cancer", yml=yaml_file, return_X_y=True, verbose=1)

# Or load the whole DataFrame (target column stays inside)
df = load_dataset("credit_card_default", yml=yaml_file, return_X_y=False, verbose=1)
```

Design notes
============
- Local-first is centralized in `load_dataset()`:
  - `file`-> look under `raw/<key>/`.
  - Other loaders (`sklearn:`, `uci:`, `kaggle-comp:`, `url:`/`http(s)`) ->
    look under `cache/<key>/` first; if not found, fetch **and persist** to `cache/<key>/`.
- Python 3.8 compatible.
"""

from __future__ import annotations

import io
import zipfile
import contextlib
import importlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Tuple, Union

import pandas as pd
import yaml


def _ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    """Ensure standard data folders exist and return their paths."""
    raw_dir = base_dir / "raw"
    cache_dir = base_dir / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "cache": cache_dir}


def _coerce_yml_and_base_dir(
    yml: Union[str, Path, Dict[str, Any]],
    base_dir: Optional[Path]
) -> Tuple[Dict[str, Any], Path]:
    """Normalize (yml, base_dir) into a (yml_dict, base_dir) pair."""
    if isinstance(yml, (str, Path)):
        yml_path = Path(yml).resolve()
        yml_dict = load_datasets_yaml(yml_path)
        use_base_dir = base_dir if base_dir is not None else yml_path.parent.resolve()
        return yml_dict, use_base_dir
    if isinstance(yml, dict):
        if base_dir is None:
            raise ValueError("base_dir must be provided when yml is a dict")
        return yml, Path(base_dir).resolve()
    raise TypeError("yml must be a str/Path or a dict")


def _persist_df_to_cache(df: pd.DataFrame, cache_dir: Path, key: str, filename: Optional[str] = None) -> Path:
    """Persist a DataFrame to cache/<key>/<filename> (CSV) and return the saved path."""
    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"{key}.csv"
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    return out_path


def _read_any(path: Path) -> pd.DataFrame:
    """Read CSV/TXT/TSV/TAB/.data/Parquet/JSON/ARFF/XLS/XLSX or a ZIP containing one of them."""
    ext = path.suffix.lower()

    # Spreadsheet formats
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(path)
        except ImportError as e:
            raise ImportError(
                "Reading Excel requires optional dependencies. "
                "Install 'openpyxl' for .xlsx and 'xlrd<2.0' for legacy .xls."
            ) from e

    # Parquet
    if ext == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise ImportError("Reading Parquet requires 'pyarrow'. Please install it.") from e

    # JSON
    if ext == ".json":
        return pd.read_json(path)

    # ARFF
    if ext == ".arff":
        # Try SciPy first
        try:
            from scipy.io import arff as scipy_arff
            data, meta = scipy_arff.loadarff(str(path))
            df = pd.DataFrame(data)
            # Decode bytes columns to str
            for c in df.columns:
                if df[c].dtype == object and len(df[c]) and isinstance(df[c].iloc[0], (bytes, bytearray)):
                    df[c] = df[c].apply(lambda b: b.decode("utf-8", errors="ignore") if isinstance(b, (bytes, bytearray)) else b)
            return df
        except ImportError:
            # Fallback to liac-arff if available
            try:
                import arff as liac_arff  # type: ignore
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    arff_obj = liac_arff.load(f)
                df = pd.DataFrame(arff_obj["data"], columns=[a[0] for a in arff_obj["attributes"]])
                return df
            except ImportError as e2:
                raise ImportError(
                    "Reading ARFF requires either 'scipy' (scipy.io.arff) or 'liac-arff'. "
                    "Please install one of them."
                ) from e2

    # TSV / TAB
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")

    # .data (UCI-style, usually CSV; fallback to whitespace)
    if ext == ".data":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=r"\s+", engine="python")

    # Plain CSV/TXT
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)

    # ZIP containing one of the supported types
    if ext == ".zip":
        with zipfile.ZipFile(path) as zf:
            # Search priority order
            candidates = [n for n in zf.namelist() if not n.endswith("/")]
            prefs = (".csv", ".tsv", ".tab", ".data", ".txt", ".parquet", ".json", ".arff", ".xlsx", ".xls")
            name = next((n for n in candidates if n.lower().endswith(prefs)), None)
            if name is None:
                raise ValueError(f"No supported tabular file found in ZIP: {path}")
            with zf.open(name) as fh:
                buf = io.BytesIO(fh.read())
                lname = name.lower()
                if lname.endswith((".xlsx", ".xls")):
                    try:
                        return pd.read_excel(buf)
                    except ImportError as e:
                        raise ImportError(
                            "Reading Excel requires optional dependencies. "
                            "Install 'openpyxl' for .xlsx and 'xlrd<2.0' for legacy .xls."
                        ) from e
                if lname.endswith(".parquet"):
                    try:
                        return pd.read_parquet(buf)
                    except ImportError as e:
                        raise ImportError("Reading Parquet requires 'pyarrow'. Please install it.") from e
                if lname.endswith(".json"):
                    return pd.read_json(buf)
                if lname.endswith(".arff"):
                    # Not supported from inside ZIP (no filename); ask to extract first
                    raise ValueError("Reading ARFF from inside ZIP is not supported; extract first.")
                if lname.endswith((".tsv", ".tab")):
                    return pd.read_csv(buf, sep="\t")
                if lname.endswith(".data"):
                    try:
                        return pd.read_csv(buf)
                    except Exception:
                        buf.seek(0)
                        return pd.read_csv(buf, sep=r"\s+", engine="python")
                # csv/txt default
                return pd.read_csv(buf)

    # Fallback: try CSV
    return pd.read_csv(path)


def _download(url: str, dest_dir: Path, save_as: Optional[str] = None) -> Path:
    import urllib.request
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = save_as or (Path(urlparse(url).path).name or "downloaded.file")
    dest = dest_dir / fname
    if not dest.exists():
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


def _find_local_path(search_dirs: List[Path]) -> Optional[Path]:
    """Return the first supported file path found under given directories."""
    exts = (".csv", ".txt", ".zip", ".parquet", ".json", ".tsv", ".tab", ".data", ".arff", ".xlsx", ".xls")
    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in exts:
                return p
    return None


def _find_local_any(search_dirs: List[Path]) -> Optional[pd.DataFrame]:
    """Try reading the first supported file found under given directories."""
    path = _find_local_path(search_dirs)
    return _read_any(path) if path else None


def _is_gzip(p: Path) -> bool:
    """Return True if file has GZIP magic header."""
    try:
        with open(p, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False

def _decompress_gzip_inplace(p: Path, verbose: int = 0) -> None:
    """
    Rename `p` -> `p.gz` and write decompressed bytes back to original path.
    Keeps a `.gz` copy next to it for reproducibility/debug.
    """
    import gzip, shutil
    gz_path = p.with_suffix(p.suffix + ".gz") if p.suffix else p.with_name(p.name + ".gz")
    try:
        p.rename(gz_path)
    except FileExistsError:
        # if gz already exists from a previous run, overwrite the plain file
        pass
    if verbose > 1:
        print(f"[rf_plateau_hpo.datasets] Detected GZIP: decompressing {gz_path.name} -> {p.name}")
    with gzip.open(gz_path, "rb") as src, open(p, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _is_zip(p: Path) -> bool:
    """Return True if file has ZIP header magic."""
    try:
        with open(p, "rb") as f:
            sig = f.read(4)
        # Common local/empty/Spanned signatures
        return sig in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    except Exception:
        return False


def _unpack_zip_disguised(p: Path, out_dir: Path, verbose: int = 0) -> None:
    """
    Unpack a file that is actually a ZIP but named with a non-.zip extension.
    We extract into out_dir. Optionally keep the original by renaming to *.zip.
    """
    import zipfile
    # Keep original as *.zip for transparency if extension isn't '.zip'
    if p.suffix.lower() != ".zip":
        zipped = p.with_suffix(p.suffix + ".zip") if p.suffix else p.with_name(p.name + ".zip")
        try:
            p.rename(zipped)
            p = zipped
        except FileExistsError:
            # If a zipped twin already exists from a previous run, just use current path
            pass

    with zipfile.ZipFile(p) as zf:
        zf.extractall(out_dir)
    if verbose > 1:
        print(f"[rf_plateau_hpo.datasets] Unpacked ZIP: {p.name} -> {out_dir}")

# ---------- UCI helpers ----------

def _try_ucimlrepo_fetch(uci_id: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Any]]:
    with contextlib.suppress(Exception):
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=uci_id)
        X, y = ds.data.features, ds.data.targets
        return X, y, ds
    return None


def _build_ucirepo_desc_mapping_if_unique(df_cols: List[str], ds: Any) -> Dict[str, str]:
    """
    Build a description-based renaming mapping only if the resulting names are unique.
    Otherwise, return an empty mapping (keep original names).
    """
    variables = getattr(ds, "variables", None)
    if variables is None or not {"name", "description"}.issubset(set(variables.columns)):
        return {}
    desc_by_name = (
        variables.set_index("name")["description"]
        .apply(lambda x: str(x).strip() if pd.notna(x) else "")
        .to_dict()
    )
    proposed = [desc_by_name.get(c, "") or c for c in df_cols]
    if len(set(proposed)) != len(proposed):
        return {}
    return {orig: new for orig, new in zip(df_cols, proposed) if orig != new}


def _rename_ucirepo_all_columns_if_unique(df: pd.DataFrame, ds: Any) -> pd.DataFrame:
    """Rename all columns using variables.description when available only if unique; else keep original names."""
    mapping = _build_ucirepo_desc_mapping_if_unique(list(df.columns), ds)
    if not mapping:
        return df
    return df.rename(columns=mapping)


def _uci_and_persist(uci_spec: str, key: str, cache_dir: Path) -> pd.DataFrame:
    """
    Fetch a UCI dataset via ucimlrepo, attempt to rename all columns to
    `variables.description` (only if unique), persist to cache/<key>/<key>.csv,
    and read it back from disk.
    """
    arg = uci_spec.split(":", 1)[1].strip()
    if not arg.isdigit():
        raise ValueError(f"Unsupported UCI spec: {uci_spec}")
    uci_id = int(arg)

    res = _try_ucimlrepo_fetch(uci_id)
    if res is None:
        raise RuntimeError(
            f"Cannot load UCI dataset id={uci_id}. "
            f"Install 'ucimlrepo' (%pip install ucimlrepo) or ensure a CSV exists under '{cache_dir / key}'."
        )
    X, y, ds = res
    if isinstance(y, pd.Series) and y.name is None:
        y = y.rename("target")
    df = pd.concat([X, y], axis=1)
    df = _rename_ucirepo_all_columns_if_unique(df, ds)
    saved = _persist_df_to_cache(df, cache_dir, key, filename=f"{key}.csv")
    return _read_any(saved)


# ---------- Kaggle competitions helper ----------

def _read_specific_file_if_exists(search_dirs: List[Path], filename: str) -> Optional[pd.DataFrame]:
    """If an exact filename exists under any of the search dirs, read and return it; else None."""
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return _read_any(candidate)
    return None

# We consider these as "supported" top-level files on Kaggle
KAGGLE_PREF_EXTS: Tuple[str, ...] = (
    ".csv", ".tsv", ".tab", ".parquet", ".json", ".xlsx", ".xls", ".data", ".zip"
)

def _kaggle_comp_pick_preferred(api, slug: str, verbose: int = 0) -> Optional[str]:
    """
    Inspect competition file list and pick a good candidate when no '@filename' is provided.
    Priority:
      1) 'train.*' with a supported extension
      2) any supported extension
      3) first file in the list
    Returns the filename (as shown by Kaggle) or None.
    """
    try:
        flist = api.competition_list_files(slug)
    except Exception:
        return None

    # Kaggle returns list of objects with attribute .name (or dicts with ['name'])
    names = []
    for f in flist:
        n = getattr(f, "name", None)
        if not n and isinstance(f, dict):
            n = f.get("name")
        if n:
            names.append(n)

    # 1) Prefer train.*
    for n in names:
        ln = n.lower()
        if ln.startswith("train.") and any(ln.endswith(ext) for ext in KAGGLE_PREF_EXTS):
            if verbose > 1:
                print(f"[rf_plateau_hpo.datasets] Picked preferred by rule #1: {n}")
            return n

    # 2) Any supported ext
    for n in names:
        ln = n.lower()
        if any(ln.endswith(ext) for ext in KAGGLE_PREF_EXTS):
            if verbose > 1:
                print(f"[rf_plateau_hpo.datasets] Picked preferred by rule #2: {n}")
            return n

    # 3) Fallback: first file if any
    if names:
        if verbose > 1:
            print(f"[rf_plateau_hpo.datasets] Picked preferred by rule #3: {names[0]}")
        return names[0]
    return None


def _kaggle_comp_and_persist(kaggle_spec: str, key: str, cache_dir: Path, verbose: int = 0) -> pd.DataFrame:
    """
    Download a Kaggle competition file using the Kaggle *Python API* (no CLI/PATH dependency),
    persist it under cache/<key>/, and return a DataFrame.

    Loader spec shape: "kaggle-comp:<slug>@<preferred_file>"
      - <slug>            e.g. "titanic"
      - <preferred_file>  e.g. "train.csv" (optional; if omitted, we auto-pick a good candidate)

    Requirements:
      - pip install kaggle
      - ~/.kaggle/kaggle.json with 600 perms and ~/.kaggle with 700 (or use KAGGLE_CONFIG_DIR)
      - Accept the competition rules in the Kaggle UI
    """
    # Parse spec "kaggle-comp:<slug>@<preferred>"
    spec = kaggle_spec.split(":", 1)[1].strip()
    parts = spec.split("@", 1)
    slug = parts[0].strip()
    # do NOT default to 'train.csv' here; we will auto-pick via API if empty
    prefer = parts[1].strip() if len(parts) == 2 and parts[1].strip() else ""

    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Local import to avoid hard dependency at module import time
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise ImportError(
            "Kaggle Python API is required for 'kaggle-comp:' loaders. "
            "Install it with: pip install kaggle"
        ) from e

    api = KaggleApi()
    try:
        api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_CONFIG_DIR
    except Exception as e:
        raise RuntimeError(
            "Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json (chmod 600) and ~/.kaggle (chmod 700), "
            "or set KAGGLE_CONFIG_DIR. Also make sure competition rules are accepted in the Kaggle UI."
        ) from e

    # Show progress only when verbose > 1
    quiet_flag = not (verbose and verbose > 1)
    if verbose > 1:
        print(f"[rf_plateau_hpo.datasets] Kaggle API: competition='{slug}', preferred='{prefer or '(auto)'}'")

    # If no explicit filename is provided, ask Kaggle for a good candidate
    if not prefer:
        try:
            flist = api.competition_list_files(slug)
        except Exception:
            flist = None

        names: List[str] = []
        if flist:
            for f in flist:
                n = getattr(f, "name", None)
                if not n and isinstance(f, dict):
                    n = f.get("name")
                if n:
                    names.append(n)

        # Supported top-level files we can handle (ZIP will be unpacked after download)
        supported_exts = (".csv", ".tsv", ".tab", ".parquet", ".json", ".xlsx", ".xls", ".data", ".zip")

        # Rule #1: train.* with supported extension
        pick = ""
        for n in names:
            ln = n.lower()
            if ln.startswith("train.") and any(ln.endswith(ext) for ext in supported_exts):
                pick = n
                if verbose > 1:
                    print(f"[rf_plateau_hpo.datasets] Auto-picked preferred (rule #1): {n}")
                break

        # Rule #2: any supported extension
        if not pick:
            for n in names:
                ln = n.lower()
                if any(ln.endswith(ext) for ext in supported_exts):
                    pick = n
                    if verbose > 1:
                        print(f"[rf_plateau_hpo.datasets] Auto-picked preferred (rule #2): {n}")
                    break

        # Rule #3: fallback to the first file (if any)
        if not pick and names:
            pick = names[0]
            if verbose > 1:
                print(f"[rf_plateau_hpo.datasets] Auto-picked preferred (rule #3): {pick}")

        prefer = pick  # may remain empty if listing failed

    # Try to download the preferred file first; otherwise, download all
    downloaded_any = False
    if prefer:
        try:
            # Saves exactly the requested top-level file into out_dir
            api.competition_download_file(slug, prefer, path=str(out_dir), force=True, quiet=quiet_flag)
            downloaded_any = True
        except Exception as e:
            if verbose:
                print(f"[rf_plateau_hpo.datasets] Preferred '{prefer}' not downloaded ({e}). Falling back to all files...")

    if not downloaded_any:
        # Downloads a zip containing all competition files
        api.competition_download_files(slug, path=str(out_dir), force=True, quiet=quiet_flag)
        downloaded_any = True

    # Normalize single preferred file if present: it may be gzipped or a disguised ZIP
    if prefer:
        target = out_dir / prefer

        # Candidate for .gz alongside target (Kaggle sometimes writes <name>.gz)
        gz_candidate = (
            target.with_suffix(target.suffix + ".gz") if target.suffix
            else target.with_name(target.name + ".gz")
        )

        if target.exists():
            # 1) GZIP-in-CSV case
            if _is_gzip(target):
                _decompress_gzip_inplace(target, verbose=verbose)
            # 2) ZIP-in-CSV case
            elif _is_zip(target):
                _unpack_zip_disguised(target, out_dir, verbose=verbose)

        elif gz_candidate.exists():
            # If Kaggle saved as *.gz, decompress to the intended name
            _decompress_gzip_inplace(gz_candidate, verbose=verbose)

    # Also scan the directory for any disguised ZIPs (no .zip extension but ZIP magic)
    for p in out_dir.iterdir():
        if p.is_file() and p.suffix.lower() not in (".zip", ".gz"):
            try:
                if _is_zip(p):
                    _unpack_zip_disguised(p, out_dir, verbose=verbose)
            except Exception:
                # best-effort; ignore oddities
                pass

    # Unpack any zip bundles we have
    for zp in out_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out_dir)
        except Exception:
            # If it's not a valid zip (rare), ignore and continue
            pass

    # If the preferred file exists after downloads, use it; otherwise pick the first supported file
    df = _read_specific_file_if_exists([out_dir], prefer) if prefer else None
    if df is None:
        df = _find_local_any([out_dir])

    if df is None:
        raise RuntimeError(
            f"Kaggle competition '{slug}' downloaded but no supported tabular file "
            f"was found in {out_dir}. Check the loader spec (preferred='{prefer}')."
        )

    return df


# ---------- Other loaders helper ----------

def _url_and_persist(url: str, key: str, cache_dir: Path) -> pd.DataFrame:
    """Download a file via URL into cache/<key>/ and read it back from disk."""
    dest_dir = cache_dir / key
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(urlparse(url).path).name or "downloaded.file"
    dest = dest_dir / fname
    if not dest.exists():
        dest = _download(url, dest_dir, fname)
    return _read_any(dest)


def _sklearn_and_persist(loader: str, target_name: Optional[str], cache_dir: Path, key: str) -> pd.DataFrame:
    """Load a scikit-learn built-in dataset, persist to cache/<key>/<key>.csv, and read it back."""
    dotted = loader.split(":", 1)[1] if loader.startswith("sklearn:") else loader
    mod_name, func_name = dotted.rsplit(".", 1)
    func = getattr(importlib.import_module(mod_name), func_name)
    bunch = func(as_frame=True)
    X, y = bunch.data, bunch.target
    if isinstance(y, pd.Series) and y.name is None:
        y = y.rename("target")
    df = pd.concat([X, y], axis=1)
    saved = _persist_df_to_cache(df, cache_dir, key, filename=f"{key}.csv")
    return _read_any(saved)


# ---------- Public API ----------

def load_datasets_yaml(yml_path: Path) -> Dict[str, Any]:
    """Load datasets registry YAML (e.g., 'data/datasets.yml')."""
    yml_path = Path(yml_path).resolve()
    with open(yml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bibtex(key: str, yml: Dict[str, Any]) -> str:
    """Return the BibTeX block for a dataset key (empty string if missing)."""
    try:
        return (yml["datasets"][key].get("bib") or "").strip()
    except Exception:
        return ""


def load_dataset(
    key: str,
    yml: Union[str, Path, Dict[str, Any]],
    base_dir: Optional[Path] = None,
    return_X_y: bool = True,
    verbose: int = 0,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Unified dataset loader.

    Local-first
    -----------
    - For `file` (or `raw`) we look for files under `raw/<key>/` (next to the YAML)
      and load them if present.
    - For all other loaders we look for files under `cache/<key>/` (next to the YAML)
      and load them if present. If not present, we fetch and persist to cache, then read.

    Supported loaders
    -----------------
    - `sklearn:` built-ins
    - `uci:<id>` via `ucimlrepo` (with description-based renaming only if unique)
    - `kaggle-comp:` via Kaggle Python API (package `kaggle`); you must accept competition rules in the Kaggle UI
    - `url:` and full http(s) URLs
    - `file` (`raw`) locally stored files

    Target handling
    ---------------
    `return_X_y=True` requires a valid `target` column specified in YAML; no inference.
    Note: the YAML `target` is used only when `return_X_y=True`. If `return_X_y=False`,
    the loader returns a DataFrame and the YAML `target` is ignored.

    Parameters
    ----------
    key : str
        Dataset key inside the YAML registry (yml['datasets'][key]).
    yml : Union[str, Path, Dict[str, Any]]
        Path to YAML or already loaded dict. If dict, base_dir must be provided.
    base_dir : Optional[Path]
        Base directory where raw/ and cache/ live. Defaults to the YAML parent when yml is a path.
    return_X_y : bool, default True
        If True, return (X, y) using the 'target' column from YAML. If False, return a DataFrame.
    verbose : int, default 0
        If 1, prints whether data were loaded from cache/file or fetched and cached.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]
        (X, y) tuple if return_X_y is True; otherwise a pandas DataFrame (with target column included).
    """
    yml_dict, base_dir_resolved = _coerce_yml_and_base_dir(yml, base_dir)
    dirs = _ensure_dirs(base_dir_resolved)

    spec = yml_dict["datasets"][key]
    loader = spec["loader"]
    target = spec.get("target")

    # Centralized local-first logic
    if loader.startswith("file") or loader.startswith("raw"):
        raw_hit = _find_local_path([dirs["raw"] / key])
        if raw_hit is not None:
            df = _read_any(raw_hit)
            if verbose:
                print(f"[rf_plateau_hpo.datasets] Loaded from raw: {raw_hit}")
    else:
        cache_hit = _find_local_path([dirs["cache"] / key])
        if cache_hit is not None:
            df = _read_any(cache_hit)
            if verbose:
                print(f"[rf_plateau_hpo.datasets] Loaded from cache: {cache_hit}")
        else:
            # Fetch and persist to cache
            if loader.startswith("sklearn:"):
                df = _sklearn_and_persist(loader, target, dirs["cache"], key)
                if verbose:
                    print(f"[rf_plateau_hpo.datasets] Fetched via sklearn and cached to: {dirs['cache'] / key / (key + '.csv')}")
            elif loader.startswith("uci:"):
                df = _uci_and_persist(loader, key, dirs["cache"])
                if verbose:
                    print(f"[rf_plateau_hpo.datasets] Fetched via UCI (ucimlrepo) and cached to: {dirs['cache'] / key / (key + '.csv')}")
            elif loader.startswith("kaggle-comp:"):
                df = _kaggle_comp_and_persist(loader, key, dirs["cache"], verbose=verbose)
                if verbose:
                    print(f"[rf_plateau_hpo.datasets] Downloaded via Kaggle competition into: {dirs['cache'] / key}")
            elif loader.startswith("url:") or loader.startswith(("http://", "https://")):
                url = loader.split(":", 1)[1] if loader.startswith("url:") else loader
                df = _url_and_persist(url, key, dirs["cache"])
                if verbose:
                    print(f"[rf_plateau_hpo.datasets] Downloaded from URL into: {dirs['cache'] / key}")
            else:
                raise ValueError(f"Unknown loader: {loader}")

    
    # Drop ignored columns (applies only after reading from disk/local file)
    ignore_spec = spec.get("ignored_columns") or spec.get("columns_ignored")
    if ignore_spec:
        if isinstance(ignore_spec, str):
            ignored = [c.strip() for c in ignore_spec.split(",") if c.strip()]
        elif isinstance(ignore_spec, list):
            ignored = [str(c).strip() for c in ignore_spec if str(c).strip()]
        else:
            ignored = []
        to_drop = [c for c in ignored if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            if verbose:
                print(f"[rf_plateau_hpo.datasets] Dropped ignored columns: {to_drop}")

    # Strict target handling
    if return_X_y:
        if not target:
            raise ValueError(
                f"YAML entry for '{key}' must define 'target' when return_X_y=True"
            )
        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found for dataset key '{key}'. "
                f"Fix 'target' in datasets.yml or ensure the loaded file includes this column. "
                f"Available columns: {list(df.columns)}"
            )
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    return df
