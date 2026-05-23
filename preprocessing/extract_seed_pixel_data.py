#!/usr/bin/env python3
"""
extract_seed_pixel_data.py
----------------------------
Extracts central seed-pixel data from waveform parquet files by selecting
pixels that pass amplitude, timing, and pulse-width requirements. The script
writes separate parquet outputs for seed pixels, cluster-size = 1,
and clusters with size greater than 1 only for 4 central pixels that we are reading out.

Usage:
    python3 extract_seed_pixel_data.py --input <path_to_parquet_file> [options]

Example:
    python3 extract_seed_pixel_data.py --input PixelPulse_merged.parquet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CENTRAL_PIXELS = {(1, 1), (1, 2), (2, 1), (2, 2)}
DT = 0.025
EXPECTED_PULSE_TIME = 75.0
PULSE_WINDOW = 25.0
MIN_SAMPLES_BELOW = 2


def detect_waveform_column(df):
    for column in df.columns:
        first_value = df[column].iloc[0]
        if isinstance(first_value, (list, np.ndarray)):
            return column
    raise RuntimeError("Waveform column not found")


def make_output_dir(parquet_file):
    input_path = Path(parquet_file)
    input_dir = input_path.parent
    run_id = input_path.stem.split("_")[-1]

    if run_id == "merged":
        output_dir = input_dir / "pdata"
    else:
        output_dir = input_dir / "pdata" / run_id

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def row_to_table(row):
    return pa.Table.from_pandas(pd.DataFrame([row]), preserve_index=False)


def efficiency(passed, total):
    return 0.0 if total == 0 else passed / total


def process(parquet_file, amp_cut=10.0):
    parquet = pq.ParquetFile(parquet_file)
    output_dir = make_output_dir(parquet_file)

    waveform_col = None
    seed_writer = None
    cs1_writer = None
    cs_gt1_writer = None

    total_pixels = 0
    valid_waveforms = 0
    pass_amp = 0
    pass_time = 0
    pass_width = 0
    final_firing = 0

    total_events = 0
    events_with_hits = 0
    central_firing = 0

    pulse_times_all = []
    pulse_times_pass = []

    print(f"Processing: {parquet_file}")
    print(f"Amplitude cut: amp > {amp_cut} mV")

    try:
        for row_group in range(parquet.num_row_groups):
            print(f"Row group {row_group + 1}/{parquet.num_row_groups}")
            df = parquet.read_row_group(row_group).to_pandas()

            if waveform_col is None:
                waveform_col = detect_waveform_column(df)

            for _, group in df.groupby("event"):
                total_events += 1
                firing_indices = []
                amplitudes = {}

                for idx, row in group.iterrows():
                    total_pixels += 1
                    waveform = row[waveform_col]

                    if not isinstance(waveform, (list, np.ndarray)):
                        continue

                    valid_waveforms += 1
                    waveform = np.asarray(waveform)

                    min_val = np.min(waveform)
                    pulse_amplitude = -min_val
                    if pulse_amplitude < abs(amp_cut):
                        continue

                    pass_amp += 1

                    min_idx = np.argmin(waveform)
                    pulse_time = min_idx * DT
                    pulse_times_all.append(pulse_time)

                    if not (EXPECTED_PULSE_TIME - PULSE_WINDOW <= pulse_time <= EXPECTED_PULSE_TIME + PULSE_WINDOW):
                        continue

                    pass_time += 1
                    pulse_times_pass.append(pulse_time)

                    below_threshold = waveform < (-amp_cut)
                    if np.sum(below_threshold) < MIN_SAMPLES_BELOW:
                        continue

                    pass_width += 1
                    firing_indices.append(idx)
                    amplitudes[idx] = pulse_amplitude
                    final_firing += 1

                if firing_indices:
                    events_with_hits += 1

                central = {
                    idx: amp for idx, amp in amplitudes.items()
                    if (group.loc[idx, "x"], group.loc[idx, "y"]) in CENTRAL_PIXELS
                }
                central_firing += len(central)

                if central:
                    seed_idx = max(central, key=central.get)
                    seed_table = row_to_table(group.loc[seed_idx])
                    if seed_writer is None:
                        seed_writer = pq.ParquetWriter(output_dir / "seed_pixels.parquet", seed_table.schema)
                    seed_writer.write_table(seed_table)

                if len(firing_indices) == 1:
                    row = group.loc[firing_indices[0]]
                    if (row["x"], row["y"]) in CENTRAL_PIXELS:
                        cs1_table = row_to_table(row)
                        if cs1_writer is None:
                            cs1_writer = pq.ParquetWriter(output_dir / "cluster_size1.parquet", cs1_table.schema)
                        cs1_writer.write_table(cs1_table)

                elif len(firing_indices) > 1:
                    for idx in firing_indices:
                        row = group.loc[idx]
                        if (row["x"], row["y"]) in CENTRAL_PIXELS:
                            csgt1_table = row_to_table(row)
                            if cs_gt1_writer is None:
                                cs_gt1_writer = pq.ParquetWriter(
                                    output_dir / "cluster_size_gt1_central.parquet",
                                    csgt1_table.schema,
                                )
                            cs_gt1_writer.write_table(csgt1_table)
    finally:
        if seed_writer is not None:
            seed_writer.close()
        if cs1_writer is not None:
            cs1_writer.close()
        if cs_gt1_writer is not None:
            cs_gt1_writer.close()

    print("========== DEBUG SUMMARY ==========")
    print(f"Total pixels: {total_pixels}")
    print(f"Valid waveform: {valid_waveforms} ({efficiency(valid_waveforms, total_pixels):.3f})")
    print(f"Pass amplitude {amp_cut} mV: {pass_amp} ({efficiency(pass_amp, valid_waveforms):.3f})")
    print(f"Pass timing: {pass_time} ({efficiency(pass_time, pass_amp):.3f})")
    print(f"Pass width: {pass_width} ({efficiency(pass_width, pass_time):.3f})")
    print(f"Final firing pixels: {final_firing} ({efficiency(final_firing, total_pixels):.3f})")

    print("--- EVENTS ---")
    print(f"Total events: {total_events}")
    print(f"Events with hits: {events_with_hits} ({efficiency(events_with_hits, total_events):.3f})")
    print(f"Central firing pixels: {central_firing} ({efficiency(central_firing, final_firing):.3f})")

    pulse_times_all = np.array(pulse_times_all)
    pulse_times_pass = np.array(pulse_times_pass)

    print("--- TIMING DEBUG ---")
    if len(pulse_times_all) > 0:
        print(f"All pulses: mean = {pulse_times_all.mean():.2f} ns, std = {pulse_times_all.std():.2f} ns")
        print(f"All pulses: min = {pulse_times_all.min():.2f}, max = {pulse_times_all.max():.2f}")

    if len(pulse_times_pass) > 0:
        print(f"Passing pulses: mean = {pulse_times_pass.mean():.2f} ns, std = {pulse_times_pass.std():.2f} ns")

    print("Data extraction complete")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--amp-cut", type=float, default=10.0, help="Minimum pulse amplitude in mV")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    process(args.input, args.amp_cut)