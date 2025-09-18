import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from group_sae.utils import MODEL_MAP, load_training_clusters, palette

FUZZ_DIR = "/home/lse/sae_training/group-sae/interp/results/fuzzing_sequential"
DETECT_DIR = "/home/lse/sae_training/group-sae/interp/results/detection_sequential"


def extract_latent(file_name: str):
    file_name = file_name.split(".txt")[0]
    layer, latent = file_name.split("_latent")
    layer = layer.split(".")[-1]
    return int(latent), int(layer)


def load_cluster_results(size, cluster_id, what="fuzzing"):
    cluster_map = load_training_clusters(size)
    layers = cluster_map[cluster_id]

    if what == "fuzzing":
        path = os.path.join(FUZZ_DIR, f"pythia-{size.lower()}/{cluster_id}")
    elif what == "detection":
        path = os.path.join(DETECT_DIR, f"pythia-{size.lower()}/{cluster_id}")
    else:
        raise ValueError("what must be either 'fuzzing' or 'detection'")

    results = {"latent": [], "layer": [], "correct": []}

    for file_name in os.listdir(path):
        latent, layer = extract_latent(file_name)

        if str(layer) in layers:
            with open(os.path.join(path, file_name), "r") as f:
                data = json.load(f)

            corrects = [prompt["correct"] for prompt in data if prompt["correct"] is not None]

            results["latent"].append(latent)
            results["layer"].append(layer)
            results["correct"].append(np.mean(corrects) * 100)

    results = pd.DataFrame(results)
    results["cluster_id"] = cluster_id
    results["size"] = size
    return results


def load_results(size, what="fuzzing"):
    training_clusters = load_training_clusters(size)
    all_results = [
        load_cluster_results(size, cluster_id, what) for cluster_id in training_clusters.keys()
    ]
    return pd.concat(all_results)


def load_g9_results(size, what="fuzzing"):
    results = load_results(size, what)
    summ_results = results.groupby(["size", "cluster_id", "layer"])["correct"].mean().reset_index()

    # Adding G=9 tag for all clusters
    summ_results["G"] = "9"

    return summ_results


# Load data for 410m model
fuzz_df = load_g9_results("410m", "fuzzing")
detect_df = load_g9_results("410m", "detection")


def plot_g9_scores(size="410m"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, layout="tight")

    n_layers = MODEL_MAP[f"pythia-{size.lower()}"]["n_layers"]
    expanded_palette = sns.blend_palette(palette, n_colors=n_layers - 2)

    # Plot fuzzing scores
    sns.barplot(
        data=fuzz_df,
        x="G",
        y="correct",
        ax=ax[0],
        zorder=2,
        color=expanded_palette[0],  # Using just one color for G=9
        legend=False,
    )

    # Plot detection scores
    sns.barplot(
        data=detect_df,
        x="G",
        y="correct",
        ax=ax[1],
        zorder=2,
        color=expanded_palette[0],  # Using just one color for G=9
        legend=False,
    )

    # Format the plots
    for i, title in enumerate(["Fuzzing", "Detection"]):
        ax[i].set_title(title, fontsize=14)
        ax[i].set_ylim(50, 80)
        ax[i].set_xlabel("G", fontsize=14)
        ax[i].set_ylabel("Score", fontsize=14)
        ax[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax[i].grid(color="#adb5bd", linestyle="--", linewidth=0.5)

    plt.suptitle(f"Pythia {size} - G=9 Performance", fontsize=16)

    # Save the figure
    plt.savefig(f"imgs/g9_scores_{size}.png", dpi=300, bbox_inches="tight")
    plt.close()


# Create and save the plot
plot_g9_scores("410m")

print("Fuzzing variance:", fuzz_df.groupby("G")["correct"].std())
print("Detection variance:", detect_df.groupby("G")["correct"].std())
# also the mean
print("Fuzzing mean:", fuzz_df.groupby("G")["correct"].mean())
print("Detection mean:", detect_df.groupby("G")["correct"].mean())
