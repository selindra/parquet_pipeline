#!/usr/bin/env python3
"""
cluster_size.py
---------------
Reads a leading-edge parquet dataset, computes per-event cluster size and seed
charge, and plots both the seed-charge histogram split by cluster size and the
overall cluster-size distribution.

Usage:
    python3 cluster_size.py <path_to_parquet_file>
    python3 cluster_size.py --pdata-dir <path_to_pdata_dir> --tag <dataset_tag>

Examples:
    python3 cluster_size.py $DIR/parquet/merged/pdata/leading_edge_all_set1.parquet
    python3 cluster_size.py --pdata-dir $DIR/parquet/merged/pdata --tag set1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# Load event-level information and build the requested summary plots.
def make_histogram(file_path):
    parquet_file = pq.ParquetFile(file_path)
    table = parquet_file.read(columns=["event", "x", "y", "charge_e"])
    df = table.to_pandas()

    cluster_size = df.groupby("event").size().rename("cluster_size")
    seed_charge = df.groupby("event")["charge_e"].max().rename("seed_charge_e")
    event_summary = pd.concat([cluster_size, seed_charge], axis=1).reset_index()

    output_dir = os.path.dirname(file_path) or "."

    colors = {
        1: "black",
        2: "#1f77b4",
        3: "#d98ab4",
        4: "#2ca25f",
        "4+": "#f0b000",
    }

    bins_charge = np.arange(
        0,
        np.ceil(event_summary["seed_charge_e"].max() / 20) * 20 + 20,
        20,
    )

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

    for cluster_size_value in [1, 2, 3, 4]:
        values = event_summary.loc[
            event_summary["cluster_size"] == cluster_size_value,
            "seed_charge_e",
        ].dropna()
        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=bins_charge,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            label=f"CS={cluster_size_value}",
            color=colors[cluster_size_value],
            edgecolor=colors[cluster_size_value],
        )

    ax.set_xlabel("Seed signal charge (e)")
    ax.set_ylabel("Normalized frequency per 20 e")
    ax.set_title("Amplitude histogram by cluster size")
    ax.set_xlim(0, 2000)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_hist = os.path.join(plots_dir, "cluster_size_histogram.png")
    plt.savefig(out_hist)
    plt.close()
    print(f"Histogram saved as {out_hist}")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    cluster_sizes = event_summary["cluster_size"].to_numpy()
    bins_cs = np.arange(0.5, cluster_sizes.max() + 1.5, 1.0)
    counts, bin_edges = np.histogram(cluster_sizes, bins=bins_cs)

    if counts.sum() > 0:
        counts_norm = counts / counts.sum()
    else:
        counts_norm = np.zeros_like(counts, dtype=float)

    ax.stairs(counts_norm, bin_edges, linewidth=2, color="black")
    ax.set_xlabel("Cluster size (pixel)")
    ax.set_ylabel("Fraction of events")
    ax.set_title("Cluster size distribution")
    ax.set_xlim(0, max(6, cluster_sizes.max() + 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dist = os.path.join(plots_dir, "cluster_size_distribution.png")
    plt.savefig(out_dist)
    plt.close()
    print(f"Cluster size distribution saved as {out_dist}")


# Resolve the input parquet file either directly or from pdata-dir plus tag.
def resolve_input_file(file_path=None, pdata_dir=None, tag="set1"):
    if file_path:
        return file_path
    if pdata_dir:
        return os.path.join(pdata_dir, f"leading_edge_all_{tag}.parquet")
    raise ValueError("Provide either a parquet file path or --pdata-dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot seed charge histogram and cluster size distribution from a parquet file"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to input parquet file",
    )
    parser.add_argument(
        "--pdata-dir",
        default=None,
        help="Directory containing leading_edge_all_<tag>.parquet",
    )
    parser.add_argument(
        "--tag",
        default="set1",
        help="Dataset tag used in the leading-edge parquet filename",
    )
    args = parser.parse_args()

    input_file = resolve_input_file(args.file, args.pdata_dir, args.tag)
    make_histogram(input_file)