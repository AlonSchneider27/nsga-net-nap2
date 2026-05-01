"""Statistical feature extraction for neural network weight/gradient tensors.

Computes 12 statistical features both globally and per-axis (axis=-1),
producing keys like "mean" and "mean_axis_-1".
"""

from functools import wraps
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import skew, kurtosis


def _axes_wrapper(
    func: Callable,
    func_name: str,
    check_for_last_axis: bool = True,
    do_flatten: bool = False,
) -> Callable:
    """Wrap a function to compute globally and along axis=-1.

    Mirrors the original _axes_wrapper from TrainTimeStatistics.py.
    do_flatten is accepted but is a no-op (matches original behavior).
    """
    @wraps(func)
    def wrapper(data: np.ndarray) -> Dict:
        result = {func_name: func(data)}
        if check_for_last_axis:
            result[f"{func_name}_axis_-1"] = func(data, axis=-1)
        return result

    return wrapper


def _axis_for_cov_flatten(data: np.ndarray) -> Dict:
    """Compute covariance globally (as variance of flattened) and per-axis.

    Uses np.cov(data.flatten()) which returns the variance of the flattened
    array (a scalar), preserving the original quirk.
    """
    data = np.asarray(data)
    result = {"covariance": np.cov(data.flatten())}
    axis_res = []
    for i in range(data.shape[-1]):
        axis_res.append(np.cov(data[..., i].flatten()))
    result["covariance_axis_-1"] = np.array(axis_res)
    return result


def _lfunc_last_axis_wrapper(func: Callable, func_name: str) -> Callable:
    """Wrap a function that needs manual iteration over the last axis.

    Used for L1 norm where we can't simply pass axis=-1.
    """
    def wrapper(data: np.ndarray) -> Dict:
        result = {func_name: func(data)}
        axis_res = []
        for i in range(data.shape[-1]):
            axis_res.append(func(data[..., i]))
        result[f"{func_name}_axis_-1"] = np.array(axis_res)
        return result

    return wrapper


# ── The 12 stats ────────────────────────────────────────────────────────────

STATS: Dict[str, Callable] = {
    "mean": _axes_wrapper(np.nanmean, "mean"),
    "variance": _axes_wrapper(np.nanvar, "variance"),
    "median": _axes_wrapper(np.nanmedian, "median"),
    "std": _axes_wrapper(np.nanstd, "std"),
    "max": _axes_wrapper(np.nanmax, "max"),
    "min": _axes_wrapper(np.nanmin, "min"),
    "covariance": _axis_for_cov_flatten,
    "skewness": _axes_wrapper(skew, "skewness", do_flatten=True),
    "kurtosis": _axes_wrapper(kurtosis, "kurtosis", do_flatten=True),
    "q-th_percentile": _axes_wrapper(
        lambda w, axis=None: np.nanpercentile(
            w, [0, 25, 75, 50, 100], axis=axis
        ).transpose(),
        func_name="q-th_percentile",
        check_for_last_axis=False,
    ),
    "L1_norm_new": _lfunc_last_axis_wrapper(
        lambda x: np.sum(np.abs(x)), "L1_norm_new"
    ),
    "L2_norm_new": _axes_wrapper(np.linalg.norm, "L2_norm_new"),
}

DEFAULT_FEATURE_MAP_STATS: List[str] = list(STATS.keys())


def extract_layer_stats(
    tensor: np.ndarray,
    stat_names: Optional[List[str]] = None,
) -> Dict:
    """Extract statistical features from a single tensor.

    Parameters
    ----------
    tensor : array-like
        The weight or gradient tensor (will be cast to float64).
    stat_names : list of str, optional
        Which stats to compute. Defaults to DEFAULT_FEATURE_MAP_STATS.

    Returns
    -------
    dict
        Flat dict, e.g. {"mean": scalar, "mean_axis_-1": array, ...}.
    """
    data = np.asarray(tensor)

    # Replace Inf with NaN before computation.
    data = np.where(np.isinf(data), np.nan, data)

    if stat_names is None:
        stat_names = DEFAULT_FEATURE_MAP_STATS

    result: Dict = {}
    for name in stat_names:
        result.update(STATS[name](data))
    return result


def extract_all_stats(
    snapshots: Dict[int, Dict[str, np.ndarray]],
    stat_names: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Dict]]:
    """Extract stats from all layers across all snapshots.

    Parameters
    ----------
    snapshots : dict
        {step_idx: {layer_name: tensor}}
    stat_names : list of str, optional
        Which stats to compute. Defaults to DEFAULT_FEATURE_MAP_STATS.

    Returns
    -------
    dict
        {step_idx: {layer_name: {stat_key: value}}}
    """
    result = {}
    for step_idx, layers in snapshots.items():
        result[step_idx] = {}
        for layer_name, tensor in layers.items():
            result[step_idx][layer_name] = extract_layer_stats(
                tensor, stat_names=stat_names
            )
    return result
