import json
import logging
import os
import argparse
import sys

import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from functools import partial
from pathlib import Path
from typing import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

from utils import CMS, count_ngrams

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Example argument parser')

    parser.add_argument('--sampled_data_path',
                        help='path for the sampled data. It should be saved using a Dataset object from datasets .save_to_disk() method and the the sentences should be stored with the "text" key')

    parser.add_argument('--seed', '-s', help='seed', default=42, type=int)

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ngram_hash_featurizer(cms: CMS, line: str) -> np.array:
    counted_ngrams = count_ngrams([line], 2)
    cms.update(counted_ngrams)
    counter = cms.counter
    cms.reset()
    return counter


def eig(A):
    M = np.mean(A.T, 1)
    C = A - M
    V = np.cov(C.T)
    return np.linalg.eig(V)[0]


def log_stats(stats, features, **kwargs):
    stat = kwargs
    cos = cosine_similarity(features, features)
    stat["max_self_sim"] = cos.max()
    stat["min_self_sim"] = cos.min()
    stat["avg_self_sim"] = cos.mean()
    eigen_values = eig(features)
    eigen_frac = eigen_values/eigen_values.sum()
    stat["eig1_frac"] = eigen_frac[0]
    stat["eig5_frac"] = eigen_frac[:5].sum()
    stats.append(stat)
    return stats


def one_pass_filter(
    stream,
    featurizer: Callable,
    out_path: str,
    cache_size: int,
    t_low: float,
    p_low: float,
    t_high: float,
    p_high: float,
    stop_idx: int,
    debug: False,
    log_step: int=1000
):
    out_path = Path(out_path)
    os.makedirs(out_path, exist_ok=True)

    out_f = open(out_path.joinpath("filtered_data.txt"), "w")
    idx = 0
    cache_features = []
    # populate initial set of cache
    bar = tqdm(total=cache_size, desc="Populating Cache")
    while idx < cache_size:
        line = next(stream)
        if isinstance(line, dict):
            line = line["text"]

        if not debug:
            out_f.write(json.dumps({"text": line})+"\n")
        feat = featurizer(line)
        cache_features.append(feat[0])
        idx += 1
        bar.update()

    cache_features = np.vstack(cache_features)
    norm_cache_features = cache_features / np.tile(np.linalg.norm(cache_features, ord=2, axis=1), (cache_features.shape[1], 1)).T
    discard_cnt, replace_cnt = 0, 0
    stats = []
    bar = tqdm(total=stop_idx, desc=f"Processing Stream (discard={discard_cnt}, replace={replace_cnt})", position=0)
    while idx < stop_idx:
        line = next(stream)
        if line is None:
            break
        if isinstance(line, dict):
            line = line["text"]

        feat = featurizer(line.strip())
        norm_feat = feat / np.sqrt(np.sum(feat**2))
        distances = 1 - np.sum(norm_feat * norm_cache_features, axis=1)
        min_val, max_val = distances.min(), distances.max()
        # if distance is so small, we have a high possibility of duplication, discard
        if min_val < t_low and np.random.rand() < p_low:
            discard_cnt += 1
            continue
        # if distance so large, we should update cache to ensure the diversity of cache
        elif max_val > t_high and np.random.rand() < p_high:
            min_idx = np.argmin(distances)
            norm_cache_features[min_idx] = norm_feat
            replace_cnt += 1

        if not debug:
            out_f.write(json.dumps({"text": line})+"\n")

        if idx % log_step == 0:
            stats = log_stats(stats, norm_cache_features, discard_cnt=discard_cnt, replace_cnt=replace_cnt, idx=idx)
        idx += 1
        bar.update()
        bar.set_description(f"Processing Stream (discard={discard_cnt}, replace={replace_cnt})")

    pd.DataFrame(stats).to_csv(out_path.joinpath("stats.csv"))
    out_f.close()


if __name__ == "__main__":
    seed=42
    cms = CMS(100, 1, seed=seed)
    featurizer = partial(ngram_hash_featurizer, cms)

    for cache_size in [100, 1000, 10000]:
        for t_low in [0.01, 0.03, 0.1, 0.3]:
            for t_high in [0.99, 0.97, 0.9, 0.7]:
                print(f"========= cache_size={cache_size}, t_low={t_low}, t_high={t_high} ========")
                data = load_dataset('the_pile', split='train', streaming=True).shuffle(seed=seed)
                one_pass_filter(
                    iter(data), featurizer,
                    out_path=f"../data/CMS100_CS{cache_size}_TL{t_low}_TH_{t_high}",
                    stop_idx=48000,  # 4800000 is all data
                    cache_size=1000,
                    t_low=0.1,
                    p_low=1.0,
                    t_high=0.9,
                    p_high=1.0,
                    debug=False,
                    log_step=100
                )
