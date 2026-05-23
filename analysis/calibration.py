#!/usr/bin/env python3
"""
calibration.py
----------------------------
Builds amplitude histograms and a simple charge calibration from extracted
central-pixel parquet files. The script plots seed and cluster amplitude
spectra, fits a double-Gaussian model to the 2x2 central pixels for
cluster-size-1 data, and derives a linear calibration in mV per electron.

Usage:
    python3 calibration.py --pdata-dir <path_to_pdata_dir>

Example:
    python3 calibration.py --pdata-dir parquet/pdata
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

CENTRAL_PIXELS = [(1, 1), (1, 2), (2, 1), (2, 2)]
BIN_WIDTH_MV = 0.5
FIT_XMIN = 80
FIT_XMAX = 105


# Basic Gaussian model used for peak fitting.
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


# Sum of two Gaussian peaks for the Fe-55 spectrum region.
def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2)


# Read amplitudes from a parquet file and split out the 2x2 central pixels.
def load_amplitudes(parquet_file):
    df = pq.read_table(parquet_file).to_pandas()

    amplitudes = []
    pixel_data = {pixel: [] for pixel in CENTRAL_PIXELS}

    waveform_col = None
    for column in df.columns:
        if isinstance(df[column].iloc[0], (list, np.ndarray)):
            waveform_col = column
            break

    if waveform_col is None:
        raise RuntimeError(f"Waveform column not found in {parquet_file}")

    for _, row in df.iterrows():
        waveform = np.asarray(row[waveform_col])
        amplitude = -np.min(waveform)
        amplitudes.append(amplitude)

        pixel = (row["x"], row["y"])
        if pixel in CENTRAL_PIXELS:
            pixel_data[pixel].append(amplitude)

    return np.asarray(amplitudes), pixel_data


# Plot one amplitude histogram, optionally normalized by bin width.
def plot_hist(data, filename, title, normalize_per_mV=False):
    if len(data) == 0:
        return

    bins = np.arange(data.min(), data.max() + BIN_WIDTH_MV, BIN_WIDTH_MV)
    counts, edges = np.histogram(data, bins=bins, density=normalize_per_mV)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ylabel = "Normalized frequency / mV" if normalize_per_mV else "Counts"

    plt.figure(figsize=(8, 6))
    plt.bar(centers, counts, width=width)
    plt.xlabel("Amplitude (mV)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)

    plt.text(
        0.97,
        0.95,
        f"N = {len(data)}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# Fit each central pixel spectrum with two Gaussians and return peak means.
def fit_pixels(pixel_data):
    mu1_list = []
    mu2_list = []
    results = {}

    for pixel, amplitudes in pixel_data.items():
        amplitudes = np.asarray(amplitudes)
        if len(amplitudes) < 50:
            continue

        bins = np.arange(amplitudes.min(), amplitudes.max() + BIN_WIDTH_MV, BIN_WIDTH_MV)
        hist, edges = np.histogram(amplitudes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])

        mask = (centers > FIT_XMIN) & (centers < FIT_XMAX)
        x = centers[mask]
        y = hist[mask]

        if len(x) == 0 or np.max(y) == 0:
            continue

        try:
            popt, _ = curve_fit(
                double_gaussian,
                x,
                y,
                p0=[np.max(y), 89, 2, np.max(y) / 6, 97, 2],
                maxfev=10000,
            )

            a1, mu1, s1, a2, mu2, s2 = popt
            if mu1 > mu2:
                mu1, mu2 = mu2, mu1
                popt = np.array([a2, mu2, s2, a1, mu1, s1])

            mu1_list.append(mu1)
            mu2_list.append(mu2)
            results[pixel] = popt

            print(f"Pixel {pixel}: mu1 = {mu1:.2f} mV, mu2 = {mu2:.2f} mV")
        except RuntimeError:
            print(f"Fit failed for pixel {pixel}")

    return np.asarray(mu1_list), np.asarray(mu2_list), results


# Plot the 2x2 central-pixel spectra and overlay fitted Gaussian components.
def plot_2x2(pixel_data, fit_results, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (pixel, amplitudes) in zip(axes, pixel_data.items()):
        amplitudes = np.asarray(amplitudes)
        if len(amplitudes) == 0:
            ax.set_title(f"Pixel {pixel}")
            continue

        bins = np.arange(amplitudes.min(), amplitudes.max() + BIN_WIDTH_MV, BIN_WIDTH_MV)
        hist, edges = np.histogram(amplitudes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        ax.bar(centers, hist, width=width, alpha=0.6)

        if pixel in fit_results:
            a1, mu1, s1, a2, mu2, s2 = fit_results[pixel]
            xfit = np.linspace(FIT_XMIN, FIT_XMAX, 400)
            ax.plot(xfit, gaussian(xfit, a1, mu1, s1), '--')
            ax.plot(xfit, gaussian(xfit, a2, mu2, s2), '--')
            ax.text(
                0.02,
                0.95,
                f"Peak 1: Î¼={mu1:.2f} mV\nPeak 2: Î¼={mu2:.2f} mV",
                fontsize=12,
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        ax.set_title(f"Pixel {pixel}", fontsize=14)
        ax.set_xlabel("Amplitude (mV)", fontsize=13)
        ax.set_ylabel("Counts", fontsize=13)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# Derive a linear calibration from the fitted Fe-55 peak positions.
def plot_calibration(mu1, mu2, plot_dir):
    if len(mu1) == 0 or len(mu2) == 0:
        print("Not enough fitted peaks to build calibration")
        return

    eh_pair_energy_ev = 3.6
    energy1_kev = 5.9
    energy2_kev = 6.5

    q1 = energy1_kev * 1000 / eh_pair_energy_ev
    q2 = energy2_kev * 1000 / eh_pair_energy_ev

    x = np.concatenate([np.full(len(mu1), q1), np.full(len(mu2), q2)])
    y = np.concatenate([mu1, mu2])

    slope = np.sum(x * y) / np.sum(x ** 2)
    gain = 1 / slope

    xfit = np.linspace(0, max(x) * 1.05, 200)
    yfit = slope * xfit

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6, label="Per-pixel peak positions")

    y1 = np.mean(mu1)
    y2 = np.mean(mu2)
    yerr1 = np.std(mu1)
    yerr2 = np.std(mu2)
    plt.errorbar(q1, y1, yerr=yerr1, fmt='o', label="5.9 keV mean")
    plt.errorbar(q2, y2, yerr=yerr2, fmt='o', label="6.5 keV mean")
    plt.plot(xfit, yfit, '--', label="Fit through origin")

    plt.text(
        0.05,
        0.80,
        f"Slope = {slope:.5f} mV/e\nGain = {gain:.2f} e/mV",
        fontsize=13,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.xlabel("Charge (electrons)", fontsize=15)
    plt.ylabel("Amplitude (mV)", fontsize=15)
    plt.title("Charge calibration", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "charge_calibration.png", dpi=200)
    plt.close()

    print(f"Calibration slope = {slope:.5f} mV/e")
    print(f"Gain = {gain:.2f} e/mV")


# Run the full plotting and calibration chain from a pdata directory.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdata-dir", required=True, help="Directory containing extracted parquet files")
    args = parser.parse_args()

    pdata_dir = Path(args.pdata_dir)
    plot_dir = pdata_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    seed, _ = load_amplitudes(pdata_dir / "seed_pixels.parquet")
    cs1, cs1_pixels = load_amplitudes(pdata_dir / "cluster_size1.parquet")
    cs_gt1, _ = load_amplitudes(pdata_dir / "cluster_size_gt1_central.parquet")
    cs_all = np.concatenate([cs1, cs_gt1])

    normalize_per_mV = True

    #to look deeper in the data:

    # plot_hist(seed, plot_dir / "seed_amp_all.png", "Seed amplitudes")
    # plot_hist(cs1, plot_dir / "cluster1_amp_norm.png", "Cluster size = 1", normalize_per_mV=normalize_per_mV)
    # plot_hist(cs1, plot_dir / "cluster1_amp.png", "Cluster size = 1")
    # plot_hist(cs_gt1, plot_dir / "cluster_gt1_amp.png", "Cluster size > 1")
    # plot_hist(cs_gt1, plot_dir / "cluster_gt1_amp_norm.png", "Cluster size > 1", normalize_per_mV=normalize_per_mV)
    # plot_hist(cs_all, plot_dir / "cluster_all_amp_norm.png", "All cluster sizes", normalize_per_mV=normalize_per_mV)

    mu1, mu2, fit_results = fit_pixels(cs1_pixels)
    plot_2x2(cs1_pixels, fit_results, plot_dir / "cluster1_pixels_2x2.png", "Cluster size = 1")
    plot_calibration(mu1, mu2, plot_dir)

    print("Plotting complete")


if __name__ == "__main__":
    main()