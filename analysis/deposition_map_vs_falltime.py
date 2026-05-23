#!/usr/bin/env python3
"""
deposition_map_vs_falltime.py
-----------------------------
Loads merged DepositedCharge data together with leading-edge timing results,
folds deposited positions into the symmetric pixel geometry, and plots 2D
histograms of deposition depth versus distance from the electrode corner for
fast and slow fall-time categories.

Usage:
    python3 deposition_map_vs_falltime.py <path_to_parquet_file> [options]

Example:
    python3 deposition_map_vs_falltime.py PixelPulse_merged.parquet --falltime_threshold 0.1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

FALLTIME_THRESHOLD = 0.10
PIXEL_PITCH = 0.01
AMP_E = 1550
BIN_UM = 0.2
SENSOR_DEPTH_UM = 25


# Load DepositedCharge data and fold positions into the symmetric pixel frame.
def load_and_filter_deposited_charge(parquet_path):
    dc_path = os.path.join(os.path.dirname(parquet_path), "DepositedCharge_merged.parquet")
    df = pd.read_parquet(dc_path)

    required_cols = [
        'event', 'dc_idx', 'charge', 'sign',
        'local_x', 'local_y', 'local_z', 'local_time'
    ]
    for column in required_cols:
        if column not in df.columns:
            raise ValueError(f"Missing required column in DepositedCharge file: {column}")

    df = df[df['sign'] == -1].copy()
    df = df.rename(
        columns={
            'local_x': 'dep_x',
            'local_y': 'dep_y',
            'local_z': 'dep_z',
            'local_time': 'dep_time',
        }
    )

    half_pitch = PIXEL_PITCH / 2
    half_pitch_um = half_pitch * 1000

    for column in ['dep_x', 'dep_y']:
        shifted = df[column] + half_pitch
        folded = shifted % PIXEL_PITCH
        df[f'{column}_folded_um'] = np.where(
            folded > half_pitch,
            (PIXEL_PITCH - folded) * 1000,
            folded * 1000,
        )

    r_from_corner = np.sqrt(df['dep_x_folded_um'] ** 2 + df['dep_y_folded_um'] ** 2)
    half_diag_um = half_pitch_um * np.sqrt(2)
    df['dep_r_um'] = half_diag_um - r_from_corner

    half_depth_mm = SENSOR_DEPTH_UM / 2 / 1000
    df['dep_z_um'] = (df['dep_z'] - half_depth_mm) * 1000

    return df, half_diag_um


# Load the leading-edge timing data and keep single-pixel events below the charge cut.
def load_and_filter_leading_edge(parquet_path):
    le_path = os.path.join(os.path.dirname(parquet_path), 'pdata/leading_edge_all_set1.parquet')
    df_le = pd.read_parquet(le_path)

    required_cols = ['event', 'x', 'y', 'charge_e', 'falltime_ns']
    for column in required_cols:
        if column not in df_le.columns:
            raise ValueError(f"Missing required column in leading edge file: {column}")

    df_le = df_le[df_le['charge_e'] <= AMP_E]
    df_le = df_le[df_le.groupby('event')['x'].transform('count') == 1]
    df_le = df_le.rename(columns={'falltime_ns': 'falltime'})

    return df_le, le_path


# Draw one 2D histogram of deposition radius versus depth.
def make_hist2d_plot(ax, df_hits, half_diag_um, title, ylabel=False):
    r_bins = np.arange(0, half_diag_um + BIN_UM, BIN_UM)
    z_bins = np.arange(-SENSOR_DEPTH_UM, BIN_UM, BIN_UM)

    hist, xedges, yedges = np.histogram2d(
        df_hits['dep_r_um'],
        df_hits['dep_z_um'],
        bins=[r_bins, z_bins],
    )

    image = ax.pcolormesh(
        xedges,
        yedges,
        hist.T,
        cmap='Blues',
        vmin=0,
        vmax=850,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)

    ax.set_xlabel(r'Distance from electrode corner (µm)', fontsize=20)
    if ylabel:
        ax.set_ylabel('Depth (µm)', fontsize=20)
        cbar = plt.colorbar(image, cax=cax)
        cbar.set_label(f'Entries (per {BIN_UM} µm × {BIN_UM} µm)', size=18)
        cbar.ax.tick_params(labelsize=18, which='major')
    else:
        cax.set_visible(False)

    ax.set_xlim(0, half_diag_um)
    ax.set_ylim(-15, 0)
    ax.set_title(title, fontsize=20, pad=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.3)


# Run the fast/slow fall-time split and save the final comparison figure.
def main():
    parser = argparse.ArgumentParser(
        description='2D histogram: distance-from-electrode vs depth, split by fall time.'
    )
    parser.add_argument('parquet_file')
    parser.add_argument(
        '--falltime_threshold',
        type=float,
        default=FALLTIME_THRESHOLD,
        help='Fall time threshold in ns (default: 0.1)',
    )
    args = parser.parse_args()

    plot_dir = os.path.join(os.path.dirname(args.parquet_file), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    df_dc, half_diag_um = load_and_filter_deposited_charge(args.parquet_file)
    print(f'Total depositions loaded: {len(df_dc)}')

    df_le, le_path = load_and_filter_leading_edge(args.parquet_file)
    valid_events = df_dc['event'].unique()
    df_le = df_le[df_le['event'].isin(valid_events)]
    print(f'After leading edge filter: {len(df_le)} hits')

    threshold = args.falltime_threshold
    fast_events = df_le[df_le['falltime'] < threshold]['event'].values
    slow_events = df_le[df_le['falltime'] >= threshold]['event'].values

    df_fast = df_dc[df_dc['event'].isin(fast_events)]
    df_slow = df_dc[df_dc['event'].isin(slow_events)]
    print(f' Fast (falltime < {threshold} ns): {len(df_fast)} depositions')
    print(f' Slow (falltime >= {threshold} ns): {len(df_slow)} depositions')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    make_hist2d_plot(
        ax1,
        df_fast,
        half_diag_um,
        title=f'Falltime < {threshold + 0.1} ns(charge = {AMP_E} e)',
    )
    make_hist2d_plot(
        ax2,
        df_slow,
        half_diag_um,
        title=f'Falltime >= {threshold + 0.1} ns(charge = {AMP_E} e)',
        ylabel=True,
    )

    plt.tight_layout()

    filename = os.path.join(plot_dir, 'depR_electrode_vs_depth_FALLTIME_split.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f'Saved: {filename}')


if __name__ == '__main__':
    main()
    