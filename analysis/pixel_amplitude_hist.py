#!/usr/bin/env python3
"""
pixel_amplitude_hist.py
----------------------------
Creates amplitude histograms from merged Parquet files for either PixelCharge
or PixelPulse data. For PixelCharge input, the script filters single-pixel
events above a charge threshold and plots the collected charge distribution.
For PixelPulse input, it detects the waveform column, extracts the minimum
pulse amplitude, applies an optional threshold cut, and saves the filtered
amplitude histogram.

Usage:
    python3 pixel_amplitude_hist.py <path_to_parquet_file> --input-type <charge|pulse> [options]

Examples:
    python3 pixel_amplitude_hist.py $DIR/parquet/merged/PixelCharge_merged.parquet --input-type charge
    python3 pixel_amplitude_hist.py $DIR/parquet/PixelCharge_extractedROOT_0.parquet --input-type charge --threshold-e 500
    python3 pixel_amplitude_hist.py $DIR/parquet/PixelPulse_extractedROOT_0.parquet --input-type pulse
    python3 pixel_amplitude_hist.py $DIR/parquet/merged/PixelPulse_merged.parquet --input-type pulse --threshold -20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_CHARGE_THRESHOLD_E = 100
DEFAULT_PULSE_THRESHOLD_MV = -10.0
DEFAULT_BINS = 100
DEFAULT_PULSE_BIN_COUNT = 300
PLOT_DIRNAME = "plots"


def detect_waveform_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if len(df[column]) == 0:
            continue
        first_value = df[column].iloc[0]
        if isinstance(first_value, (list, np.ndarray)):
            return column
    raise RuntimeError("Waveform column not found")


def ensure_plot_dir(input_path: Path) -> Path:
    plot_dir = input_path.parent / PLOT_DIRNAME
    plot_dir.mkdir(exist_ok=True)
    return plot_dir


def load_and_filter_charge(parquet_path: Path, threshold_e: float) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    required_cols = ["event", "x", "y", "abs_q"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["abs_q"] >= threshold_e]
    df = df[df.groupby("event")["abs_q"].transform("count") == 1]
    return df


def plot_charge_histogram(amplitudes: np.ndarray, bins: int, output_file: Path, threshold_e: float) -> None:
    hist_color = "#984EA3"

    plt.figure(figsize=(8, 5.5))
    plt.hist(
        amplitudes,
        bins=bins,
        color=hist_color,
        alpha=0.85,
        edgecolor="none",
    )
    plt.xlabel("Collected charge (e)", fontsize=17)
    plt.ylabel("Counts", fontsize=17)
    plt.title(f"Single cluster total charge distribution, Threshold = {int(threshold_e)} e", fontsize=17)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.ylim(0, plt.ylim()[1] * 1.1)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def process_charge(file_path: Path, bins: int, threshold_e: float) -> Path:
    df = load_and_filter_charge(file_path, threshold_e)
    if df.empty:
        raise RuntimeError("No charge entries remain after filtering")

    plot_dir = ensure_plot_dir(file_path)
    output_file = plot_dir / f"pixel_charge_spectrum_{file_path.stem}.png"
    plot_charge_histogram(df["abs_q"].to_numpy(), bins=bins, output_file=output_file, threshold_e=threshold_e)
    return output_file


def pulse_range_pass(parquet_file: Path):
    pf = pq.ParquetFile(parquet_file)
    waveform_col = None
    global_min = np.inf
    global_max = -np.inf

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if waveform_col is None:
            waveform_col = detect_waveform_column(df)

        for wf in df[waveform_col]:
            if not isinstance(wf, (list, np.ndarray)):
                continue
            min_amp = np.min(wf)
            global_min = min(global_min, min_amp)
            global_max = max(global_max, min_amp)

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("No valid waveforms found")

    return pf, waveform_col, global_min, global_max


def pulse_hist_pass(pf: pq.ParquetFile, waveform_col: str, bins: np.ndarray, threshold_mv: float):
    total = 0
    below_thr = 0
    hist_cut = np.zeros(len(bins) - 1)

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        for wf in df[waveform_col]:
            if not isinstance(wf, (list, np.ndarray)):
                continue
            wf = np.asarray(wf)
            min_amp = np.min(wf)
            total += 1
            if min_amp < threshold_mv:
                below_thr += 1
                hist_cut += np.histogram(min_amp, bins=bins)[0]

    if total == 0:
        raise RuntimeError("No valid waveforms found")

    return hist_cut, total, below_thr


def plot_pulse_histogram(
    hist_cut: np.ndarray,
    bins: np.ndarray,
    threshold_mv: float,
    total: int,
    below_thr: int,
    output_file: Path,
) -> None:
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    filtered_total = np.sum(hist_cut)

    if filtered_total > 0:
        mean_f = np.sum(bin_centers * hist_cut) / filtered_total
        std_f = np.sqrt(np.sum(hist_cut * (bin_centers - mean_f) ** 2) / filtered_total)
    else:
        mean_f = 0.0
        std_f = 0.0

    fraction = below_thr / total if total else 0.0

    plt.figure(figsize=(8, 5.5))
    plt.step(bin_centers, hist_cut, where="mid")
    plt.xlabel("Amplitude (mV)", fontsize=17)
    plt.ylabel("Counts", fontsize=17)
    plt.title(f"Amplitude Histogram (Threshold =  {threshold_mv} mV)", fontsize=17)
    plt.grid(True)

    text_box = (
        f"Entries = {int(filtered_total)}\n"
        f"Mean = {mean_f:.2f} mV\n"
        f"Std = {std_f:.2f} mV\n"
        f"Fraction = {fraction:.3f}"
    )
    plt.text(
        0.72,
        0.95,
        text_box,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def process_pulse(file_path: Path, threshold_mv: float, bin_count: int) -> Path:
    pf, waveform_col, global_min, global_max = pulse_range_pass(file_path)
    margin = 1.0
    bins = np.linspace(global_min - margin, global_max + margin, bin_count)
    hist_cut, total, below_thr = pulse_hist_pass(pf, waveform_col, bins, threshold_mv)

    plot_dir = ensure_plot_dir(file_path)
    output_file = plot_dir / f"pixel_pulse_min_amp_filtered_{file_path.stem}.png"
    plot_pulse_histogram(hist_cut, bins, threshold_mv, total, below_thr, output_file)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create amplitude histograms from PixelCharge or PixelPulse parquet files."
    )
    parser.add_argument("parquet_file", help="Input parquet file path")
    parser.add_argument(
        "--input-type",
        required=True,
        choices=["charge", "pulse"],
        help="Type of parquet content to process",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for charge histogram mode",
    )
    parser.add_argument(
        "--threshold-e",
        type=float,
        default=DEFAULT_CHARGE_THRESHOLD_E,
        help="Minimum charge threshold in electrons for charge mode",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PULSE_THRESHOLD_MV,
        help="Pulse threshold in mV for pulse mode",
    )
    parser.add_argument(
        "--pulse-bins",
        type=int,
        default=DEFAULT_PULSE_BIN_COUNT,
        help="Number of bin edges for pulse histogram mode",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    file_path = Path(args.parquet_file)

    if args.input_type == "charge":
        output = process_charge(file_path, bins=args.bins, threshold_e=args.threshold_e)
    else:
        output = process_pulse(file_path, threshold_mv=args.threshold, bin_count=args.pulse_bins)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()