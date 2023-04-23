import os
from typing import *

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


feat2short = {
    "cache size": "CS",
    "duplication threshold": "TL",
    "duplication discard probability": "PL",
    "update cache threshold": "TH",
    "update cache probability": "PH",
    "bucket size": "CMS"
}

short2feat = {v: k for k, v in feat2short.items()}


def get_param_from_path(data_dir):
    p_list = str(data_dir).split("/")[-1].split("_")
    out = {}
    for p in p_list:
        for short, feat in short2feat.items():
            if p.startswith(short):
                out[feat] = float(p.replace(short, ""))
    return out


def tensor_str_to_float(s: str):
    return float(s.split(",")[0].replace("tensor(", ""))


def load_dir(data_dir) -> pd.DataFrame:
    if data_dir.joinpath("stats.csv").exists():
        df = pd.read_csv(f"{data_dir}/stats.csv")
        for col in ["min_self_sim", "max_self_sim", "avg_self_sim"]:
            if df[col].dtype != float:
                if "tensor(" in str(df[col][0]):
                    df[col] = df[col].apply(lambda x: tensor_str_to_float(s))
                else:
                    raise NotImplementedError()
        feat = get_param_from_path(data_dir)
        for k, v in feat.items():
            df[k] = v
    else:
        print(f"stats file does not exist for {data_dir}, skipping ...")
        df = pd.DataFrame()
    return df


def make_plot(df, out_dir:Path, x, y, hue, max_idx: Optional[int]):
    print(f"making plot {y}_vs_{x}_across_{hue}, max_idx={max_idx}...")
    plt.figure()
    max_idx = max(df.idx) if max_idx is None else max_idx
    g = sns.lineplot(data=df[df.idx <= max_idx], x=x, y=y, hue=hue)
    plt.title(f"{y} across different {hue} over filtering")
    plt.xlabel(x if x != "idx" else "Number of filtered datapoints")
    plt.ylabel(y)
    g.set(ylim=(0, None))
    out_path = out_dir.joinpath(f"{y}_vs_{x}_across_{hue}.png")
    if max_idx is not None:
        out_path = str(out_path).replace(".png", f"_{max_idx}.png")
    plt.savefig(out_path)
    plt.close()


def plot_dirs(dirs: List[Path], max_idx: int):

    df = pd.concat([load_dir(d) for d in dirs])
    out_dir = dirs[0].parents[1].joinpath("dump/figures")
    os.makedirs(out_dir, exist_ok=True)
    make_plot(df, out_dir, "idx", "avg_self_sim", "cache size", max_idx)
    # make_plot(df, out_dir, "idx", "avg_self_sim", "duplication threshold", max_idx)
    # make_plot(df, out_dir, "idx", "avg_self_sim", "update cache probability", max_idx)


if __name__ == "__main__":
    max_idx = 100000
    plot_dirs(list(Path(__file__).parents[1].joinpath("data").glob("CMS*TH0.99_*")), max_idx)
    # plot_dirs(list(Path(__file__).parents[1].joinpath("data").glob("CMS*")), max_idx)
