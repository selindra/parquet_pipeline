#!/usr/bin/env python3
"""
leading_edge_plot.py
--------------------
Reads leading-edge parquet datasets produced by leading_edge_extract.py
and creates charge-versus-fall-time density plots for cluster-size-1,
cluster-size-greater-than-1, and combined datasets.

Usage:
    python3 leading_edge_plot.py --pdata-dir <path_to_pdata_dir>

Example:
    python3 leading_edge_plot.py --pdata-dir $DIR/parquet/merged/pdata/
"""

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

SHIFT_NS = 0.1


# Plot a normalized 2D charge-versus-time density map.
def plot_2d(ax, charge, time, title, set_xlabel=True):
    mask = np.isfinite(time)
    charge = charge[mask]
    time = time[mask] + SHIFT_NS

    xbin = 20
    ybin = 0.025

    xb = np.arange(charge.min(), charge.max() + xbin, xbin)
    yb = np.arange(time.min(), time.max() + ybin, ybin)

    hist, xedges, yedges = np.histogram2d(charge, time, bins=[xb, yb])
    bin_area = xbin * ybin
    hist = hist / (hist.sum() * bin_area)

    image = ax.pcolormesh(
        xedges,
        yedges,
        hist.T,
        norm=LogNorm(vmin=0.5e-4, vmax=1e-1),
    )

    if set_xlabel:
        ax.set_xlabel(r"Seed signal charge ($e^{-}$)", fontsize=17)

    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=4, width=1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set_title(title, fontsize=20)
    return image


# Load charge, timing, event, and pixel data from one parquet dataset.
def load_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    charge = df["charge_e"].to_numpy(dtype=float)
    time = df["falltime_ns"].to_numpy(dtype=float)
    events = df["event"].to_numpy()
    pixels = df[["x", "y"]].to_numpy()
    return charge, time, events, pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot leading-edge charge vs fall-time from parquet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdata-dir",
        required=True,
        help="Directory containing leading_edge_*.parquet files",
    )
    parser.add_argument(
        "--tag",
        default="set1",
        help="Dataset tag used when files were saved",
    )
    args = parser.parse_args()

    pdata = args.pdata_dir
    tag = args.tag

    cs1_path = os.path.join(pdata, f"leading_edge_cs1_{tag}.parquet")
    csgt1_path = os.path.join(pdata, f"leading_edge_csgt1_{tag}.parquet")

    charge1, time1, events1, pixels1 = load_dataset(cs1_path)
    charge2, time2, events2, pixels2 = load_dataset(csgt1_path)

    mask1 = np.isfinite(time1)
    mask2 = np.isfinite(time2)

    charge_all = np.concatenate([charge1[mask1], charge2[mask2]])
    time_all = np.concatenate([time1[mask1], time2[mask2]])

    fig = plt.figure(figsize=(6, 10))
    grid = gridspec.GridSpec(
        2,
        2,
        width_ratios=[20, 1],
        height_ratios=[1, 1],
        wspace=0.16,
        hspace=0.1,
    )

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[:, 1])

    ax1.text(
        -0.18,
        0.45,
        "CS = 1",
        transform=ax1.transAxes,
        rotation=90,
        va="bottom",
        ha="center",
        fontsize=20,
        weight="bold",
    )
    ax2.text(
        -0.18,
        0.45,
        "CS > 1",
        transform=ax2.transAxes,
        rotation=90,
        va="bottom",
        ha="center",
        fontsize=20,
        weight="bold",
    )

    image1 = plot_2d(ax1, charge1, time1, "", set_xlabel=False)
    image2 = plot_2d(ax2, charge2, time2, "", set_xlabel=False)

    colorbar = fig.colorbar(image2, cax=cax)
    colorbar.set_label(r"Normalized frequency (per 20 $e^{-}$ x 25 ps)", size=18)
    colorbar.ax.tick_params(labelsize=20, width=1.5, length=6)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.minorticks_off()
    ax2.minorticks_off()


    plots_dir = os.path.join(pdata, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out1 = os.path.join(plots_dir, f"seed_time_vs_charge_split_{tag}.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out1)

    fig2, ax = plt.subplots(figsize=(8, 7))
    image = plot_2d(ax, charge_all, time_all, "All cluster sizes")
    colorbar2 = fig2.colorbar(image, ax=ax)
    colorbar2.set_label(r"Normalized frequency (per 20 $e^{-}$ x 25 ps)", size=18)
    colorbar2.ax.tick_params(labelsize=20, width=1.5, length=6 )
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.minorticks_off()

    
    fig2.text(
        0.01,
        0.5,
        "Seed signal leading edge time (ns)",
        va="center",
        rotation="vertical",
        fontsize=17,
    )

    out2 = os.path.join(plots_dir, f"seed_time_vs_charge_all_{tag}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out2)