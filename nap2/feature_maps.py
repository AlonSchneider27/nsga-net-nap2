"""Feature map creation for the NAP2 autoencoder.

Converts per-layer statistics into fixed-size numpy arrays of shape
[max_layers, values_per_feature, num_stats] (default [65, 100, 12]).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from nap2.stats import DEFAULT_FEATURE_MAP_STATS


def log_normalize(x: np.ndarray) -> np.ndarray:
    """Apply log normalization: log(1 + |x|) * sign(x).

    Dataset-agnostic transform that compresses the dynamic range of feature
    maps without requiring per-dataset statistics.  Enables cross-dataset
    transfer (KT ~0.728 vs raw 0.521).
    """
    return np.log1p(np.abs(x)) * np.sign(x)


RELEVANT_LAYER_NAMES = (
    "conv2d/kernel",
    "dense/kernel",
    "dense/bias",
    "stem.0.weight",
)


def is_layer_relevant(layer_name: str) -> bool:
    """Check whether a layer name matches any of the relevant layer patterns."""
    for name in RELEVANT_LAYER_NAMES:
        if name in layer_name:
            return True
    return False


def _group_stats_to_feature(
    layer_stats: Dict, feature_name: str
) -> List:
    """Collect all stat values whose key starts with feature_name.

    Keys containing 'orig' are excluded. Each matching value is flattened
    into a list. The global value (exact match on feature_name) comes first,
    followed by axis variants.
    """
    values = []
    # Global value first (exact key match)
    if feature_name in layer_stats and "orig" not in feature_name:
        val = np.array(layer_stats[feature_name], dtype="object").flatten()
        values.append(val)

    # Then axis-specific keys
    for key in layer_stats:
        if key == feature_name:
            continue
        if key.startswith(feature_name) and "orig" not in key:
            val = np.array(layer_stats[key], dtype="object").flatten()
            values.append(val)

    return values


def _flatten_and_pad(
    collected_values: List, values_per_feature: int
) -> np.ndarray:
    """Flatten collected values and pad/truncate to fixed size."""
    flat = []
    for val in collected_values:
        if isinstance(val, np.ndarray):
            flat.extend(val.flatten().tolist())
        elif isinstance(val, list):
            flat.extend(val)
        else:
            flat.append(val)

    if len(flat) >= values_per_feature:
        return np.array(flat[:values_per_feature], dtype=np.float64)
    else:
        result = np.zeros(values_per_feature, dtype=np.float64)
        result[: len(flat)] = flat
        return result


def create_feature_map(
    layer_stats: Dict[str, Dict],
    max_layers: int = 65,
    values_per_feature: int = 100,
    stat_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Create a feature map from per-layer statistics.

    Parameters
    ----------
    layer_stats : dict
        {layer_name: {stat_key: value, ...}, ...}
    max_layers : int
        Number of layers in output (padded/truncated). Default 65.
    values_per_feature : int
        Number of values per feature. Default 100.
    stat_names : list of str, optional
        Which stat prefixes to use as features. Defaults to
        DEFAULT_FEATURE_MAP_STATS (12 stats).

    Returns
    -------
    np.ndarray
        Shape [max_layers, values_per_feature, len(stat_names)].
    """
    if stat_names is None:
        stat_names = DEFAULT_FEATURE_MAP_STATS

    num_features = len(stat_names)
    step_features = []

    for layer_name, stats in layer_stats.items():
        if not is_layer_relevant(layer_name):
            continue

        feature_arrays = []
        for feature_name in stat_names:
            collected = _group_stats_to_feature(stats, feature_name)
            padded = _flatten_and_pad(collected, values_per_feature)
            feature_arrays.append(padded)

        # Stack features: shape (num_features, values_per_feature)
        layer_map = np.stack(feature_arrays, axis=0)
        step_features.append(layer_map)

    if len(step_features) == 0:
        return np.zeros((max_layers, values_per_feature, num_features), dtype=np.float64)

    # Stack layers: shape (num_layers, num_features, values_per_feature)
    arr = np.stack(step_features, axis=0)
    # Swap to (num_layers, values_per_feature, num_features)
    arr = np.swapaxes(arr, 1, 2)

    # Pad or truncate layers
    num_layers = arr.shape[0]
    if num_layers > max_layers:
        arr = arr[:max_layers]
    elif num_layers < max_layers:
        padding = np.zeros(
            (max_layers - num_layers, values_per_feature, num_features),
            dtype=np.float64,
        )
        arr = np.concatenate([arr, padding], axis=0)

    return arr


def create_feature_map_sequence(
    all_stats: Dict[int, Dict[str, Dict]],
    **kwargs,
) -> np.ndarray:
    """Create feature maps for a sequence of snapshots.

    Parameters
    ----------
    all_stats : dict
        {step_idx: {layer_name: {stat_key: value, ...}, ...}, ...}
    **kwargs
        Forwarded to create_feature_map (max_layers, values_per_feature,
        stat_names).

    Returns
    -------
    np.ndarray
        Shape [N, max_layers, values_per_feature, num_features].
    """
    maps = []
    for step_idx in sorted(all_stats.keys()):
        feature_map = create_feature_map(all_stats[step_idx], **kwargs)
        maps.append(feature_map)
    return np.stack(maps, axis=0)
