#!/usr/bin/env python3
"""
merge_parquet_files.py
----------------------------
Merges parquet files produced from extracted ROOT data into one file per data
prefix and renumbers the event column so event IDs remain continuous across
all merged input files. Files are grouped by the part of the filename before
"_extractedROOT_", and merged outputs are written into a "merged" subfolder.

Usage:
    python3 merge_parquet.py <input_dir> [options]

Examples:
    python3 merge_parquet.py $DIR/parquet/
    parquet_data_analysis]$ python3 merge_parquet.py $DIR/parquet/ --filter DepositedCharge --event-offset 1000
    
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_SUFFIX = "_merged.parquet"
MATCH_TOKEN = "_extractedROOT_"


def group_files(input_dir: str, filter_name: Optional[str] = None) -> Dict[str, List[str]]:
    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    groups: Dict[str, List[str]] = {}

    for file_path in files:
        name = os.path.basename(file_path)

        if MATCH_TOKEN not in name:
            continue

        if filter_name and filter_name not in name:
            continue

        prefix = name.split(MATCH_TOKEN)[0]
        groups.setdefault(prefix, []).append(file_path)

    return groups


def numeric_sort_key(file_path: str):
    stem = Path(file_path).stem
    tail = stem.split("_")[-1]
    try:
        return int(tail)
    except ValueError:
        return tail


def shift_event_column(table: pa.Table, event_offset: int):
    if "event" not in table.schema.names:
        return table, -1

    event_col = table["event"].to_numpy()
    if len(event_col) == 0:
        return table, -1

    shifted = pa.array(event_col + event_offset)
    updated = table.set_column(
        table.schema.get_field_index("event"),
        "event",
        shifted,
    )
    return updated, int(event_col.max())


def merge_parquet_fast(
    prefix: str,
    files: List[str],
    input_dir: str,
    event_offset: int = 0,
) -> Path:
    print(f"Merging {prefix} ({len(files)} files)")

    files = sorted(files, key=numeric_sort_key)

    output_dir = Path(input_dir) / "merged"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{prefix}{OUTPUT_SUFFIX}"

    writer = None
    current_offset = event_offset

    try:
        for file_path in files:
            print(f"  reading: {os.path.basename(file_path)}")
            parquet_file = pq.ParquetFile(file_path)
            max_event_in_file = -1

            for batch in parquet_file.iter_batches():
                table = pa.Table.from_batches([batch])
                table, batch_max_event = shift_event_column(table, current_offset)
                max_event_in_file = max(max_event_in_file, batch_max_event)

                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)

                writer.write_table(table)

            if max_event_in_file >= 0:
                current_offset += max_event_in_file + 1

            print(f"  new event offset = {current_offset}")
    finally:
        if writer is not None:
            writer.close()

    print(f"  -> saved: {output_file}\n")
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge parquet files with continuous event numbering"
    )
    parser.add_argument("input_dir", help="Folder containing parquet files")
    parser.add_argument(
        "--event-offset",
        type=int,
        default=0,
        help="Starting event offset for all matching files (default: 0)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only merge files whose name contains this string",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    groups = group_files(args.input_dir, args.filter)

    if not groups:
        print("No matching parquet files found")
        return

    for prefix, files in groups.items():
        merge_parquet_fast(
            prefix=prefix,
            files=files,
            input_dir=args.input_dir,
            event_offset=args.event_offset,
        )


if __name__ == "__main__":
    main()
