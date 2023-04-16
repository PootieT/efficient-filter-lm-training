from typing import *
import numpy as np


def one_pass_filter(
    stream,
    featurizer: Callable,
    out_path: str,
    cache_size: int,
    t_low: float,
    p_low: float,
    t_high: float,
    p_high: float,
    stop_idx: int
):
    out_f = open(out_path, "w")
    idx = 0
    cache_features = []
    while idx < cache_size:
        line = next(stream)
        out_f.write(line)
        feat = featurizer(line)
        cache_features.append(feat)
        idx += 0

    cache_features = np.vstack(cache_features)
    norm_cache_features = cache_features / np.sqrt(np.sum(cache_features, axis=1))
    while idx < stop_idx:
        line = next(stream)
        if line is None:
            break

        feat = featurizer(line.strip())
        norm_feat = feat / np.sqrt(np.sum(feat))
        distances = 1 - np.sum(norm_feat * norm_cache_features, axis=1)
        min_val = min(distances)
        if min_val < t_low and np.random.rand() < p_low:
            continue
        elif min_val > t_high and np.random.rand() > p_high:
            min_idx = np.argmin(distances)
            norm_cache_features[min_idx] = norm_feat
        out_f.write(line)



