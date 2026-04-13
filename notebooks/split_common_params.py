from typing import Any, Dict, List, Tuple, Sequence
import numpy as np


def _is_simple_immutable(x: Any) -> bool:
    return isinstance(
        x,
        (type(None), bool, int, float, str, bytes, np.generic),
    )


def _values_same(a: Any, b: Any) -> bool:
    # Use equality for simple immutable scalars; use identity for everything else.
    if _is_simple_immutable(a) and _is_simple_immutable(b):
        return a == b
    return a is b


def split_common_params(
    configs: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract key-value pairs that appear in every dict in configs with the same value,
    and return (common_params, param_list) where each element of param_list has those keys removed.

    For large or mutable objects (arrays, callables, dicts, lists), "same value" is identity (is).
    For simple immutable scalars, "same value" is equality (==).
    """
    if not configs:
        return {}, []

    common_keys = set(configs[0].keys())
    for d in configs[1:]:
        common_keys.intersection_update(d.keys())

    common_params: Dict[str, Any] = {}
    for k in sorted(common_keys):
        v0 = configs[0][k]
        if all(_values_same(v0, d[k]) for d in configs[1:]):
            common_params[k] = v0

    param_list: List[Dict[str, Any]] = []
    for d in configs:
        param_list.append({k: v for k, v in d.items() if k not in common_params})

    return common_params, param_list