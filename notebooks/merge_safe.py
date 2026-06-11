"""
merge_safe.py
-------------

Conflict-aware dictionary merging for experiment configuration dictionaries.

The helper is intentionally conservative: when the same key appears in both
input dictionaries with different values, it raises by default. This prevents
silent overwrites when composing shared and per-run parameters.
"""
import warnings
from typing import Any, Dict

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _is_simple_immutable(x: Any) -> bool:
    if np is not None:
        numpy_scalar = isinstance(x, getattr(np, "generic", ()))
    else:
        numpy_scalar = False
    return isinstance(x, (type(None), bool, int, float, str, bytes)) or numpy_scalar


def _values_same(a: Any, b: Any) -> bool:
    if _is_simple_immutable(a) and _is_simple_immutable(b):
        return a == b
    return a is b


def merge_safe(
    dst: Dict[str, Any],
    src: Dict[str, Any],
    *,
    on_conflict: str = "error",
    context: str = "",
) -> Dict[str, Any]:
    mode = on_conflict.lower().strip()
    if mode not in {"ok", "info", "warn", "error"}:
        raise ValueError("on_conflict must be one of: 'ok', 'info', 'warn', 'error'.")

    prefix = f"[{context}] " if context else ""
    for k, v in src.items():
        if k in dst and not _values_same(dst[k], v):
            msg = f"{prefix}key conflict: '{k}'"
            if mode == "error":
                raise KeyError(msg)
            if mode == "warn":
                warnings.warn(msg, RuntimeWarning)
            elif mode == "info":
                print(msg)
        dst[k] = v
    return dst
