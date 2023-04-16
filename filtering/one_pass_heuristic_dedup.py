import json
import logging
import os
import argparse
import pdb
import sys

import pandas as pd
import torch
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
    parser = argparse.ArgumentParser(description="Parser for conducing one pass dataset filter")

    parser.add_argument("--seed", "-s", help="seed", default=42, type=int)
    parser.add_argument("--bucket-size", help="CMS hash bucket size", type=int, default=10000)
    parser.add_argument("--cache-size", help="cache size", default=1000, type=int)
    parser.add_argument(
        "--t-low",
        help="low threshold below which data will be filtered out with a probability",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--t-high",
        help="high threshold above which data will be added to the cache, replacing a point"
        "closest to the data point",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--p-low", help="probability of removing data after exceeding threshold", default=1.0, type=float
    )
    parser.add_argument(
        "--p-high", help="probability of adding data to cache after exceeding threshold", default=1.0, type=float
    )
    parser.add_argument(
        "--stop-idx", help="maximum amount of data to filter", default=4800000, type=int
    )
    parser.add_argument(
        "--log-step", help="number of steps between each log", default=1000, type=int
    )
    parser.add_argument(
        "--debug", help="if debug, do not write any filtered text", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return args


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
    # cos = cosine_similarity(features, features)
    cos = torch.nn.CosineSimilarity(dim=-1)(features.unsqueeze(1), features.unsqueeze(0))
    stat["max_self_sim"] = cos.max()
    stat["min_self_sim"] = cos.min()
    stat["avg_self_sim"] = cos.mean()
    # these take forever for big matrices
    # eigen_values = eig(features)
    # eigen_frac = eigen_values / eigen_values.sum()
    # stat["eig1_frac"] = eigen_frac[0]
    # stat["eig5_frac"] = eigen_frac[:5].sum()
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
    log_step: int = 1000,
):
    out_path = Path(out_path)
    os.makedirs(out_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
            out_f.write(json.dumps({"text": line}) + "\n")
        feat = featurizer(line)
        cache_features.append(feat[0])
        idx += 1
        bar.update()

    cache_features = np.vstack(cache_features)
    norm_cache_features = torch.tensor((
        cache_features / np.tile(np.linalg.norm(cache_features, ord=2, axis=1), (cache_features.shape[1], 1)).T
    )).to(device)
    discard_cnt, replace_cnt = 0, 0
    stats = []
    bar = tqdm(total=stop_idx, desc=f"Processing Stream (discard={discard_cnt}, replace={replace_cnt})")#, position=0)
    while idx < stop_idx:
        line = next(stream)
        if line is None:
            break
        if isinstance(line, dict):
            line = line["text"]

        feat = torch.tensor(featurizer(line.strip()), device=device)
        norm_feat = feat / torch.sqrt(torch.sum(feat ** 2))
        distances = 1 - torch.sum(norm_feat * norm_cache_features, dim=1)
        min_val, max_val = distances.min(), distances.max()

        # if distance is so small, we have a high possibility of duplication, discard
        if min_val < t_low and np.random.rand() < p_low:
            discard_cnt += 1
            continue

        # if distance so large, we should update cache to ensure the diversity of cache
        elif max_val > t_high and np.random.rand() < p_high:
            min_idx = torch.argmin(distances)
            norm_cache_features[min_idx] = norm_feat
            replace_cnt += 1

        if not debug:
            out_f.write(json.dumps({"text": line}) + "\n")

        if idx % log_step == 0:
            stats = log_stats(stats, norm_cache_features, discard_cnt=discard_cnt, replace_cnt=replace_cnt, idx=idx)
        idx += 1
        bar.update()
        bar.set_description(f"Processing Stream (discard={discard_cnt}, replace={replace_cnt}, "
                            f"min-d={min_val:.3f}, max-d={max_val:.3f})")

    pd.DataFrame(stats).to_csv(out_path.joinpath("stats.csv"))
    out_f.close()


def main(args):
    cms = CMS(args.bucket_size, 1, seed=args.seed)
    featurizer = partial(ngram_hash_featurizer, cms)

    print(f"========= cache_size={args.cache_size}, t_low={args.t_low}, t_high={args.t_high} ========")
    out_path = f"../data/CMS{args.bucket_size}_CS{args.cache_size}_TL{args.t_low}_TH{args.t_high}"
    # if os.path.exists(out_path):
    #     print(f"path exist. Skipping {out_path}")
    #     exit(0)

    data = load_dataset("the_pile", split="train", streaming=True).shuffle(seed=args.seed)
    with torch.no_grad():
        one_pass_filter(
            iter(data),
            featurizer,
            out_path=out_path,
            stop_idx=args.stop_idx,  # 4800000 is all data
            cache_size=args.cache_size,
            t_low=args.t_low,
            p_low=args.p_low,
            t_high=args.t_high,
            p_high=args.p_high,
            debug=args.debug,
            log_step=args.log_step,
        )


if __name__ == "__main__":
    # seed = 42
    # cms = CMS(100, 1, seed=seed)
    # featurizer = partial(ngram_hash_featurizer, cms)
    #
    # for cache_size in [100, 1000, 10000]:
    #     for t_low in [0.01, 0.03, 0.1, 0.3]:
    #         for t_high in [0.99, 0.97, 0.9, 0.7]:
    #             print(f"========= cache_size={cache_size}, t_low={t_low}, t_high={t_high} ========")
    #             data = load_dataset("the_pile", split="train", streaming=True).shuffle(seed=seed)
    #             one_pass_filter(
    #                 iter(data),
    #                 featurizer,
    #                 out_path=f"../data/CMS100_CS{cache_size}_TL{t_low}_TH{t_high}",
    #                 stop_idx=48000,  # 4800000 is all data
    #                 cache_size=cache_size,
    #                 t_low=t_low,
    #                 p_low=1.0,
    #                 t_high=t_high,
    #                 p_high=1.0,
    #                 debug=False,
    #                 log_step=100,
    #             )
    args = parse_args()
    main(args)
