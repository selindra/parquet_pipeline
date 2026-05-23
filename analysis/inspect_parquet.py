#!/usr/bin/env python3
"""
inspect_parquet.py
----------------------------
Inspects a parquet file produced in the analysis workflow and prints basic
metadata, event-range information, coordinate ranges, and a small row preview.
The script is intended for quick sanity checks on large HEP parquet datasets,
especially to verify event numbering and x/y pixel coverage.

Usage:
    python3 inspect_parquet.py <path_to_parquet_file> [options]

Example:
    python3 inspect_parquet.py data.parquet --preview-events 5
"""

import argparse

import numpy as np
import pyarrow.parquet as pq


# Read file-level information and inspect event numbering across row groups.
def inspect_parquet(file_path, preview_events=10):
    parquet_file = pq.ParquetFile(file_path)

    print("=== BASIC FILE INFO ===")
    print(f"File: {file_path}")
    print(f"Row groups: {parquet_file.num_row_groups}")
    print(f"Columns: {parquet_file.schema.names}")
    print(f"Total rows: {parquet_file.metadata.num_rows}")

    if "event" not in parquet_file.schema.names:
        print("No 'event' column found.")
        return

    required_columns = ["event"]
    has_x = "x" in parquet_file.schema.names
    has_y = "y" in parquet_file.schema.names
    if has_x:
        required_columns.append("x")
    if has_y:
        required_columns.append("y")

    global_min = np.inf
    global_max = -np.inf
    unique_event_count = 0
    repeated_rows = 0
    last_event_prev_chunk = None

    first_events = []
    last_events_buffer = []
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=required_columns)
        events = table.column("event").to_numpy()

        if len(events) == 0:
            continue

        global_min = min(global_min, events.min())
        global_max = max(global_max, events.max())

        if has_x:
            x = table.column("x").to_numpy()
            x_min = min(x_min, x.min())
            x_max = max(x_max, x.max())

        if has_y:
            y = table.column("y").to_numpy()
            y_min = min(y_min, y.min())
            y_max = max(y_max, y.max())

        if len(first_events) < preview_events:
            needed = preview_events - len(first_events)
            first_events.extend(events[:needed].tolist())

        last_events_buffer.extend(events.tolist())
        if len(last_events_buffer) > preview_events:
            last_events_buffer = last_events_buffer[-preview_events:]

        repeated_rows += len(events) - len(np.unique(events))

        change_points = np.empty(len(events), dtype=bool)
        change_points[0] = events[0] != last_event_prev_chunk
        if len(events) > 1:
            change_points[1:] = events[1:] != events[:-1]

        unique_event_count += np.count_nonzero(change_points)
        last_event_prev_chunk = events[-1]

    print("=== TRUE EVENT RANGE ===")   
    print(f"Event min: {int(global_min)}")
    print(f"Event max: {int(global_max)}")
    print(f"Unique event count: {unique_event_count}")
    print(f"Non-unique rows due to repeated pixels/events: {repeated_rows}")

    print("=== EVENT PREVIEW ===")
    print(f"First events: {first_events}")
    print(f"Last events: {last_events_buffer}")

    if has_x or has_y:
        print("=== COORDINATE RANGE ===")
        if has_x:
            print(f"x min: {int(x_min)}")
            print(f"x max: {int(x_max)}")
        if has_y:
            print(f"y min: {int(y_min)}")
            print(f"y max: {int(y_max)}")

    print("=== DATA PREVIEW (first 10 rows) ===")
    shown_rows = 0
    rows_to_show = 10

    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group)
        available = table.num_rows
        take = min(available, rows_to_show - shown_rows)
        if take <= 0:
            break
        shown_rows += take
        print(table.slice(0, take).to_pandas())
        if shown_rows >= rows_to_show:
            break

    print("NOTE:")
    print("Row offset is not event offset when one event has multiple pixel rows.")
    print("Unique event count is computed from event transitions across row groups.")


# Parse command-line options for parquet inspection.
def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect parquet files used in the analysis workflow"
    )
    parser.add_argument("file", help="Parquet file path")
    parser.add_argument(
        "--preview-events",
        type=int,
        default=5,
        help="Number of event values to preview from the start and end",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    inspect_parquet(args.file, preview_events=args.preview_events)