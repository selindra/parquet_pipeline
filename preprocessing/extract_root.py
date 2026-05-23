#!/usr/bin/env python3
"""
extract_root.py
----------------------------
Extracts Allpix-squared ROOTObjectWriter output into Parquet files for
selected object types such as MCTrack, MCParticle, PixelCharge, PixelPulse,
PixelHit, and DepositedCharge. The script reads ROOT trees, flattens their
contents into tabular form, and writes one Parquet file per object type for
faster downstream analysis.

Usage:
    python3 extract_root.py <path_to_root_file> <detector_name> [options]

Examples:
    python3 extract_root.py data.root mydetector
    python3 extract_root.py $DIR/Tree.root APTS_OPAMP --multiple_files  --process_batch --batch_size 1 --batch_index 0

"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import ROOT
import pyarrow as pa
import pyarrow.parquet as pq


def prepare_tree(tree, branch_name):
    if not tree:
        return
    tree.SetBranchStatus('*', 0)
    tree.SetBranchStatus(branch_name, 1)
    tree.SetCacheSize(50_000_000)
    tree.AddBranchToCache(branch_name, True)


def get_n_events(tree, max_events):
    if max_events is None:
        return tree.GetEntries()
    return min(tree.GetEntries(), int(max_events))


def ensure_output_dir(path_like):
    out_dir = Path(path_like)
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def write_rows(rows, schema_arrays, names, out_path, writer):
    if not rows:
        return writer, rows

    table = pa.Table.from_arrays(
        [builder(rows) for builder in schema_arrays],
        names=names,
    )

    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)

    writer.write_table(table)
    rows.clear()
    return writer, rows


def extract_mctrack(root_file, cfg, file_suffix):
    tree = root_file.Get('MCTrack')
    if not tree:
        return

    t0 = time.time()
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"MCTrack_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, 'global')
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int32()),
        lambda r: pa.array([x[2] for x in r], type=pa.float32()),
        lambda r: pa.array([x[3] for x in r], type=pa.float32()),
        lambda r: pa.array([x[4] for x in r], type=pa.float32()),
        lambda r: pa.array([x[5] for x in r], type=pa.float32()),
    ]
    names = ['event', 'track_idx', 'kin_E_initial', 'tot_E_initial', 'kin_E_final', 'tot_E_final']

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch('global')
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for idx_t, track in enumerate(vec):
                rows.append((
                    event,
                    idx_t,
                    track.getKineticEnergyInitial(),
                    track.getTotalEnergyInitial(),
                    track.getKineticEnergyFinal(),
                    track.getTotalEnergyFinal(),
                ))
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"MCTrack -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def extract_mcparticle(root_file, cfg, file_suffix):
    tree = root_file.Get('MCParticle')
    if not tree:
        return

    t0 = time.time()
    det = cfg['detector']
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"MCParticle_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, det)
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int32()),
        lambda r: pa.array([x[2] for x in r], type=pa.float32()),
        lambda r: pa.array([x[3] for x in r], type=pa.float32()),
        lambda r: pa.array([x[4] for x in r], type=pa.float32()),
        lambda r: pa.array([x[5] for x in r], type=pa.float32()),
        lambda r: pa.array([x[6] for x in r], type=pa.float32()),
        lambda r: pa.array([x[7] for x in r], type=pa.float32()),
        lambda r: pa.array([x[8] for x in r], type=pa.float32()),
        lambda r: pa.array([x[9] for x in r], type=pa.float32()),
        lambda r: pa.array([x[10] for x in r], type=pa.float32()),
        lambda r: pa.array([x[11] for x in r], type=pa.float32()),
        lambda r: pa.array([x[12] for x in r], type=pa.float32()),
        lambda r: pa.array([x[13] for x in r], type=pa.float32()),
        lambda r: pa.array([x[14] for x in r], type=pa.float32()),
        lambda r: pa.array([x[15] for x in r], type=pa.float32()),
        lambda r: pa.array([x[16] for x in r], type=pa.float32()),
        lambda r: pa.array([x[17] for x in r], type=pa.float32()),
    ]
    names = [
        'event', 'particle_idx',
        'loc_start_x', 'loc_start_y', 'loc_start_z',
        'loc_end_x', 'loc_end_y', 'loc_end_z',
        'glob_start_x', 'glob_start_y', 'glob_start_z',
        'glob_end_x', 'glob_end_y', 'glob_end_z',
        'ref_x', 'ref_y', 'ref_z', 'time'
    ]

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch(det)
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for idx_p, particle in enumerate(vec):
                ls = particle.getLocalStartPoint()
                le = particle.getLocalEndPoint()
                gs = particle.getGlobalStartPoint()
                ge = particle.getGlobalEndPoint()
                rp = particle.getLocalReferencePoint()
                rows.append((
                    event, idx_p,
                    ls.x(), ls.y(), ls.z(),
                    le.x(), le.y(), le.z(),
                    gs.x(), gs.y(), gs.z(),
                    ge.x(), ge.y(), ge.z(),
                    rp.x(), rp.y(), rp.z(),
                    particle.getLocalTime(),
                ))
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"MCParticle -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def extract_pixelcharge(root_file, cfg, file_suffix):
    tree = root_file.Get('PixelCharge')
    if not tree:
        return

    t0 = time.time()
    det = cfg['detector']
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"PixelCharge_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, det)
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int16()),
        lambda r: pa.array([x[2] for x in r], type=pa.int16()),
        lambda r: pa.array([x[3] for x in r], type=pa.float32()),
        lambda r: pa.array([x[4] for x in r], type=pa.int16()),
        lambda r: pa.array([x[5] for x in r], type=pa.list_(pa.float32())),
    ]
    names = ['event', 'x', 'y', 'abs_q', 'pulse_size', 'charges']

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch(det)
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for charge in vec:
                try:
                    idx = charge.getPixel().getIndex()
                    px = int(idx.x())
                    py = int(idx.y())
                    abs_q = charge.getAbsoluteCharge()
                    vals = []
                    pulse = charge.getPulse()
                    if pulse:
                        n = pulse.size()
                        vals = [pulse.at(i) for i in range(n)]
                    else:
                        n = 0
                    rows.append((event, px, py, abs_q, n, vals))
                except Exception:
                    continue
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"PixelCharge -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def extract_pixelpulse(root_file, cfg, file_suffix):
    tree = root_file.Get('PixelPulse')
    if not tree:
        return

    t0 = time.time()
    det = cfg['detector']
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"PixelPulse_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, det)
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int16()),
        lambda r: pa.array([x[2] for x in r], type=pa.int16()),
        lambda r: pa.array([x[3] for x in r], type=pa.int32()),
        lambda r: pa.array([x[4] for x in r], type=pa.list_(pa.float32())),
    ]
    names = ['event', 'x', 'y', 'pulse_size', 'waveform_mV']

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch(det)
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for pulse in vec:
                try:
                    idx = pulse.getPixel().getIndex()
                    px = int(idx.x())
                    py = int(idx.y())
                    n = pulse.size()
                    vals = [pulse.at(i) * 1e9 for i in range(n)]
                    rows.append((event, px, py, n, vals))
                except Exception:
                    continue
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"PixelPulse -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def extract_pixelhit(root_file, cfg, file_suffix):
    tree = root_file.Get('PixelHit')
    if not tree:
        return

    t0 = time.time()
    det = cfg['detector']
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"PixelHit_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, det)
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int16()),
        lambda r: pa.array([x[2] for x in r], type=pa.int16()),
        lambda r: pa.array([x[3] for x in r], type=pa.float64()),
    ]
    names = ['event', 'x', 'y', 'signal_mV']

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch(det)
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for hit in vec:
                try:
                    idx = hit.getPixel().getIndex()
                    px = int(idx.x())
                    py = int(idx.y())
                    rows.append((event, px, py, abs(hit.getSignal() * 1e9)))
                except Exception:
                    continue
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"PixelHit -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def extract_depositedcharge(root_file, cfg, file_suffix):
    tree = root_file.Get('DepositedCharge')
    if not tree:
        return

    t0 = time.time()
    det = cfg['detector']
    out_dir = ensure_output_dir(cfg['output_dir'])
    out_path = out_dir / f"DepositedCharge_{cfg['output_name']}{file_suffix}.parquet"
    rows = []
    flush_every = cfg['row_flush']
    n_events = get_n_events(tree, cfg['max_events'])
    prepare_tree(tree, det)
    writer = None

    schema_arrays = [
        lambda r: pa.array([x[0] for x in r], type=pa.int32()),
        lambda r: pa.array([x[1] for x in r], type=pa.int32()),
        lambda r: pa.array([x[2] for x in r], type=pa.int32()),
        lambda r: pa.array([x[3] for x in r], type=pa.int8()),
        lambda r: pa.array([x[4] for x in r], type=pa.float32()),
        lambda r: pa.array([x[5] for x in r], type=pa.float32()),
        lambda r: pa.array([x[6] for x in r], type=pa.float32()),
        lambda r: pa.array([x[7] for x in r], type=pa.float64()),
    ]
    names = ['event', 'dc_idx', 'charge', 'sign', 'local_x', 'local_y', 'local_z', 'local_time']

    try:
        for event in range(n_events):
            tree.GetEntry(event)
            branch = tree.GetBranch(det)
            if not branch:
                continue
            vec = getattr(tree, branch.GetName())
            for idx_dc, dc in enumerate(vec):
                lp = dc.getLocalPosition()
                rows.append((
                    event,
                    idx_dc,
                    dc.getCharge(),
                    dc.getSign(),
                    lp.x(),
                    lp.y(),
                    lp.z(),
                    dc.getLocalTime(),
                ))
            if len(rows) >= flush_every:
                writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
        writer, rows = write_rows(rows, schema_arrays, names, out_path, writer)
    finally:
        if writer is not None:
            writer.close()

    print(f"DepositedCharge -> {out_path} ({n_events} events) in {time.time() - t0:.2f}s")


def dump_all(
    cfg,
    file_suffix="",
    MCTrack=True,
    MCParticle=True,
    PixelCharge=True,
    PixelPulse=True,
    PixelHit=True,
    DepositedCharge=True,
):
    ROOT.gSystem.Load(cfg['libAllpixObjects'])

    try:
        root_file = ROOT.TFile.Open(cfg['root_file'], 'READ')
    except Exception:
        print(f"Error: could not open ROOT file {cfg['root_file']}")
        return

    if not root_file:
        print(f"Error: could not open ROOT file {cfg['root_file']}")
        return

    try:
        if MCTrack:
            extract_mctrack(root_file, cfg, file_suffix)
        if MCParticle:
            extract_mcparticle(root_file, cfg, file_suffix)
        if PixelCharge:
            extract_pixelcharge(root_file, cfg, file_suffix)
        if PixelPulse:
            extract_pixelpulse(root_file, cfg, file_suffix)
        if PixelHit:
            extract_pixelhit(root_file, cfg, file_suffix)
        if DepositedCharge:
            extract_depositedcharge(root_file, cfg, file_suffix)
    finally:
        root_file.Close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract Allpix2 ROOTObjectWriter objects into Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'rootfile',
        metavar='rootfile',
        type=str,
        help='Path to the ROOT file to process. In multiple file mode, pass the common base path.',
    )
    parser.add_argument(
        'detector',
        metavar='detector',
        type=str,
        help='Simulated detector name (for example detector1 or dut).',
    )
    parser.add_argument(
        '--multiple_files',
        action='store_true',
        help='Process multiple ROOT files with a similar basename in the same folder.',
    )
    parser.add_argument(
        '--process_batch',
        action='store_true',
        help='Process only a batch of files in multiple file mode.',
    )
    parser.add_argument(
        '--batch_size', '-bs',
        type=int,
        default=10,
        help='Number of files in a batch.',
    )
    parser.add_argument(
        '--batch_index', '-bi',
        type=int,
        default=None,
        help='Batch index to process, starting from 0.',
    )
    parser.add_argument(
        '-l',
        metavar='libAllpixObjects',
        default='/data/alice/mselina/allpix_installation/allpix-squared/lib/libAllpixObjects.so',
        help='Path to the libAllpixObjects library.',
    )
    parser.add_argument(
        '--output_directory', '-o',
        default='parquet',
        help='Directory for output files. If ".", save next to the input file.',
    )
    return parser


def build_config(args):
    root_path = Path(args.rootfile)
    return {
        'root_file': args.rootfile,
        'detector': args.detector,
        'output_dir': root_path.parent.joinpath(args.output_directory),
        'libAllpixObjects': args.l,
        'output_name': 'extractedROOT',
        'max_events': None,
        'row_flush': 50000,
    }


def collect_root_files(root_path):
    folder = root_path.parent
    base = root_path.stem
    base_clean = re.sub(r'[_\-]?\d+$', '', base)

    return sorted(
        folder.glob(f"{base_clean}*.root"),
        key=lambda x: int(re.search(r'(\d+)(?=\.root$)', x.name).group()),
    )


def main():
    args = build_parser().parse_args()
    root_path = Path(args.rootfile)
    config = build_config(args)

    if args.multiple_files:
        root_files = collect_root_files(root_path)

        if not root_files:
            print('No matching ROOT files found')
            sys.exit(1)

        if args.process_batch:
            if args.batch_index is None:
                raise ValueError('Batch index must be provided with --process_batch')
            start = args.batch_index
            stop = start + args.batch_size
            root_files = root_files[start:stop]

        print(f"Processing {len(root_files)} ROOT files")

        for root_file in root_files:
            print(f"--- Processing {root_file} ---")
            config['root_file'] = str(root_file)
            match = re.search(r'(\d+)(?=\.root$)', root_file.name)
            suffix = f"_{match.group(1)}" if match else ''
            dump_all(config, file_suffix=suffix)
    else:
        dump_all(config)


if __name__ == '__main__':
    main()