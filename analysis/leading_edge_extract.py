#!/usr/bin/env python3
"""
leading_edge_extract.py
-----------------------
Reads cluster_size1.parquet and cluster_size_gt1_central.parquet produced by
extract_seed_pixel_data.py, applies the CFD waveform-processing chain,
and saves leading-edge timing datasets in parquet format for downstream plots.

Output columns: event, x, y, charge_e, falltime_ns

Usage:
    python3 leading_edge_extract.py --pdata-dir <path_to_pdata_dir>

Example:
    python3 leading_edge_extract.py --pdata-dir $DIR/parquet/merged/pdata/
"""

import argparse
import os
from pathlib import Path


import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys
sys.path.append(str(Path(__file__).resolve().parent / "../utils"))
import waveform_analysis as wa


CONVERSION_FACTOR = 18.52  # electrons per mV
DT = 0.025  # ns per sample
T_START = 50.0  # ns

BASELINE_EVAL_NS = 2.5
BASELINE_FIRST_BEFORE = 15.0
UNDERLINE_EVAL_NS = 1.25
UNDERLINE_FIRST_AFTER = 21.5
WINDOW_BEFORE_T0_NS = 25.0
WINDOW_AFTER_T0_NS = 30.0
CFD_XMIN_BEFORE_T0 = 15.0
CFD_XMAX_AFTER_T0 = 21.5


# Detect the waveform column automatically from array-like parquet fields.
def _detect_waveform_col(df):
    for column in df.columns:
        value = df[column].iloc[0]
        if isinstance(value, (list, np.ndarray)):
            return column
    raise RuntimeError('No waveform column found in parquet file.')


# Detect x/y pixel-coordinate columns using common naming conventions.
def _detect_pixel_cols(df):
    px_col, py_col = None, None
    for column in df.columns:
        name = column.lower()
        if name in ('pixel_x', 'pix_x', 'col', 'x'):
            px_col = column
        if name in ('pixel_y', 'pix_y', 'row', 'y'):
            py_col = column
    return px_col, py_col


# Apply the Set-1 CFD chain and extract charge and fall-time information.
def set1_process_waveform(waveform, dt=DT, min_amplitude_mv=5.5):
    wf = np.asarray(waveform, dtype=float)
    time = np.arange(len(wf)) * dt + T_START
    graph = (time, wf)

    try:
        if not wa.hasSignal(graph, threshold=min_amplitude_mv):
            return None
    except Exception:
        return None

    t0bin = wa.FindLeftNearBaseline(graph, cut=7, pointsWithinCut=10, totalStep=10)
    if t0bin < 0:
        return None

    left_offset = int(round(WINDOW_BEFORE_T0_NS / dt))
    right_offset = int(round(WINDOW_AFTER_T0_NS / dt))

    if t0bin - left_offset < 0 or t0bin + right_offset >= len(time):
        return None

    window = slice(t0bin - left_offset, t0bin + right_offset)
    graph_local = (time[window], wf[window])

    t0bin_local = wa.FindLeftNearBaseline(
        graph_local,
        cut=7,
        pointsWithinCut=10,
        totalStep=10,
    )
    if t0bin_local < 0:
        return None

    baseline_ok = t0bin_local - int(round((BASELINE_FIRST_BEFORE + BASELINE_EVAL_NS) / dt)) >= 0
    underline_ok = t0bin_local + int(round((UNDERLINE_FIRST_AFTER + UNDERLINE_EVAL_NS) / dt)) < len(graph_local[0])
    if not (baseline_ok and underline_ok):
        return None

    baseline, _, _ = wa.GetDefaultBaseline(
        graph_local,
        t0_bin=t0bin_local,
        dt_ns=dt,
        evaluation_time_ns=BASELINE_EVAL_NS,
        start_time_before_t0_ns=BASELINE_FIRST_BEFORE + BASELINE_EVAL_NS,
    )
    underline, _, _ = wa.GetDefaultUnderline(
        graph_local,
        t0_bin=t0bin_local,
        dt_ns=dt,
        evaluation_time_ns=UNDERLINE_EVAL_NS,
        start_time_after_t0_ns=UNDERLINE_FIRST_AFTER,
    )

    t0 = graph_local[0][t0bin_local]
    xmin = t0 - CFD_XMIN_BEFORE_T0
    xmax = t0 + CFD_XMAX_AFTER_T0

    if not (
        xmin - BASELINE_EVAL_NS > graph_local[0][0]
        and xmax + UNDERLINE_EVAL_NS < graph_local[0][-1]
        and baseline - underline > min_amplitude_mv
    ):
        return None

    cfd_times = []
    for pos, frac_pct in enumerate(range(10, 100, 10)):
        level = baseline - (frac_pct / 100.0) * (baseline - underline)
        backward = frac_pct <= 50

        tcfd = wa.FindOnGraph(
            graph_local,
            level,
            xmin,
            xmax,
            interpolate=1,
            dx=dt,
            backw=backward,
        )

        if pos > 0 and (tcfd == -999.0 or tcfd < cfd_times[pos - 1]):
            idx = np.abs(graph_local[0] - cfd_times[pos - 1]).argmin() + 1
            tcfd = graph_local[0][idx] if idx < len(graph_local[0]) else np.nan

        if tcfd == -999.0:
            tcfd = np.nan

        cfd_times.append(tcfd)

    cfd_times = np.asarray(cfd_times, dtype=float)
    t10 = cfd_times[0]
    t50 = cfd_times[4]
    falltime = (t50 - t10) if (np.isfinite(t10) and np.isfinite(t50)) else np.nan

    return {
        'charge_e': (baseline - underline) * CONVERSION_FACTOR,
        'falltime_ns': falltime,
        'baseline': baseline,
        'underline': underline,
        't0': t0,
        'cfd_times': cfd_times,
    }


# Extract charge and fall-time values from one parquet waveform file.
def extract_charge_time(parquet_file, min_charge_e=400.0, min_amp_mv=5.5):
    print(f"Reading: {parquet_file}")
    parquet = pq.ParquetFile(parquet_file)

    charges = []
    times = []
    event_ids = []
    pixel_ids = []
    raw_amp = []
    accepted_amp = []
    failed_amp = []

    waveform_col = None
    px_col = None
    py_col = None

    total = 0
    timed = 0
    cfd_fail = 0
    amp_reject = 0

    for row_group in range(parquet.num_row_groups):
        df = parquet.read_row_group(row_group).to_pandas()

        if waveform_col is None:
            waveform_col = _detect_waveform_col(df)
            px_col, py_col = _detect_pixel_cols(df)

        for _, row in df.iterrows():
            total += 1
            waveform = row[waveform_col]
            if not isinstance(waveform, (list, np.ndarray)):
                continue

            result = set1_process_waveform(waveform, dt=DT, min_amplitude_mv=min_amp_mv)
            if result is None:
                continue

            charge = result['charge_e']
            raw_amp.append(charge)

            if charge < min_charge_e:
                amp_reject += 1
                continue

            falltime = result['falltime_ns']
            if not np.isfinite(falltime):
                cfd_fail += 1
                failed_amp.append(charge)
                continue

            timed += 1
            accepted_amp.append(charge)
            charges.append(charge)
            times.append(falltime)
            event_ids.append(row['event'])

            if px_col is not None and py_col is not None:
                pixel_ids.append((row[px_col], row[py_col]))
            else:
                pixel_ids.append((-1, -1))

        del df

    raw_amp = np.asarray(raw_amp)
    accepted_amp = np.asarray(accepted_amp)
    failed_amp = np.asarray(failed_amp)

    print('--- SUMMARY ---')
    print(f' Total rows: {total}')
    print(f' Signal found: {len(raw_amp)}')
    print(f' Passed amp cut: {len(raw_amp) - amp_reject}')
    print(f' CFD failures: {cfd_fail}')
    print(f' Good timing: {timed}')

    return (
        np.asarray(charges),
        np.asarray(times),
        np.asarray(event_ids),
        np.asarray(pixel_ids),
        raw_amp,
        accepted_amp,
        failed_amp,
    )


# Save the extracted timing dataset in parquet format.
def save_leading_edge_dataset(out_path, charge, time, event, pixel):
    n = min(len(charge), len(time), len(event), len(pixel))
    charge = np.asarray(charge[:n])
    time = np.asarray(time[:n])
    event = np.asarray(event[:n])
    pixel = np.asarray(pixel[:n])

    mask = np.isfinite(time)
    df = pd.DataFrame({
        'event': event[mask],
        'x': pixel[mask, 0],
        'y': pixel[mask, 1],
        'charge_e': charge[mask],
        'falltime_ns': time[mask],
    })
    df.to_parquet(out_path, index=False)
    print(f'Saved: {out_path} ({len(df)} entries)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set-1 CFD extraction from parquet waveform files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--pdata-dir',
        required=True,
        help='Directory containing cluster_size1.parquet and cluster_size_gt1_central.parquet',
    )
    parser.add_argument(
        '--min-charge-e',
        type=float,
        default=400.0,
        help='Minimum charge cut in electrons',
    )
    parser.add_argument(
        '--min-amp-mv',
        type=float,
        default=5.5,
        help='Minimum signal amplitude for hasSignal() in mV',
    )
    parser.add_argument(
        '--output-tag',
        default='set1',
        help='Tag appended to output filename',
    )
    args = parser.parse_args()

    pdata = args.pdata_dir
    tag = args.output_tag

    cs1_file = os.path.join(pdata, 'cluster_size1.parquet')
    csgt1_file = os.path.join(pdata, 'cluster_size_gt1_central.parquet')

    charge1, time1, events1, pixels1, raw1, acc1, fail1 = extract_charge_time(
        cs1_file,
        args.min_charge_e,
        args.min_amp_mv,
    )
    charge2, time2, events2, pixels2, raw2, acc2, fail2 = extract_charge_time(
        csgt1_file,
        args.min_charge_e,
        args.min_amp_mv,
    )

    save_leading_edge_dataset(
        os.path.join(pdata, f'leading_edge_all_{tag}.parquet'),
        np.concatenate([charge1, charge2]),
        np.concatenate([time1, time2]),
        np.concatenate([events1, events2]),
        np.concatenate([pixels1, pixels2]),
    )
    save_leading_edge_dataset(
        os.path.join(pdata, f'leading_edge_cs1_{tag}.parquet'),
        charge1,
        time1,
        events1,
        pixels1,
    )
    save_leading_edge_dataset(
        os.path.join(pdata, f'leading_edge_csgt1_{tag}.parquet'),
        charge2,
        time2,
        events2,
        pixels2,
    )