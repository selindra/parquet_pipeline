# Allpix simulation analysis scripts

Small Python and Bash utilities for converting ROOT simulation output to Parquet format, extracting waveform-derived observables, and producing analysis plots for APTS pixel Fe-55 simulation studies. The workflow covers ROOT extraction, Parquet files inspection, leading-edge timing extraction, charge calibration, and several focused plotting helpers for qualitative data assessment. This repository accompanies the doctoral thesis of Mariia Selina and contains the analysis scripts used for the results described there.

## What is in the repo

The analysis works on several ROOT objects exported to Parquet, including `MCParticle`, `PixelCharge`, `PixelPulse`, `PixelHit`, and `DepositedCharge`.

## Repository layout

```text
.
├── preprocessing/
│   ├── extract_root.py
│   ├── merge_parquet.py
│   └── extract_seed_pixel_data.py
├── analysis/
│   ├── calibration.py
│   ├── cluster_size.py
│   ├── deposition_map_vs_falltime.py
│   ├── inspect_parquet.py
│   ├── leading_edge_extract.py
│   ├── leading_edge_plot.py
│   ├── pixel_amplitude_hist.py
│   └── plot_waveforms_range.py
├── run_full_pipeline.sh
├──LICENSE.txt
├──.gitignore
└── README.md
```

## Workflow

`run_full_pipeline.sh` contains the full ready-to-run pipeline; the main steps are summarized below.

### 1. Preprocessing

Convert the ROOT file to parquet, merge the parquet chunks, and extract waveform subsets for later timing work.

```bash
python3 preprocessing/extract_root.py <run_dir>/Tree.root APTS_OPAMP --multiple_files
python3 preprocessing/merge_parquet.py <run_dir>/parquet/
python3 preprocessing/extract_seed_pixel_data.py --input <run_dir>/parquet/merged/PixelPulse_merged.parquet
```

### 2. Quick inspection

Use the helper scripts to inspect file structure and make quick amplitude or waveform checks before running the heavier analysis chain.

```bash
python3 analysis/inspect_parquet.py <run_dir>/parquet/merged/PixelPulse_merged.parquet
python3 analysis/pixel_amplitude_hist.py <run_dir>/parquet/merged/PixelCharge_merged.parquet --input-type charge
python3 analysis/pixel_amplitude_hist.py <run_dir>/parquet/merged/PixelPulse_merged.parquet --input-type pulse
python3 analysis/plot_waveforms_range.py --file <run_dir>/parquet/merged/pdata/cluster_size1.parquet
```

- `inspect_parquet.py` prints row-group metadata, event ranges, coordinate ranges, and a small preview to sanity-check parquet content.
- `pixel_amplitude_hist.py` plots the histograms of signals amplitude for both mV from PixelPulse object and charge from PixelCharge `--input-type charge|pulse`.
- `plot_waveforms_range.py` selects and plots a number of waveforms in a chosen amplitude range after converting amplitude from mV to electrons.

### 3. Calibration

`calibration.py` fits a double-Gaussian model to the 2x2 central pixels for cluster-size = 1 events, and derives a linear charge calibration in mV per electron.

```bash
python3 analysis/calibration.py --pdata-dir <run_dir>/parquet/merged/pdata/
```

### 4. Leading-edge timing

`leading_edge_extract.py` processes `cluster_size1.parquet` and `cluster_size_gt1_central.parquet`, applies the CFD chain, calculates CFD 10-50%, and writes tagged parquet outputs such as `leading_edge_all_set1.parquet` with columns `event`, `x`, `y`, `charge_e`, and `falltime_ns`.

```bash
python3 analysis/leading_edge_extract.py --pdata-dir <run_dir>/parquet/merged/pdata/ --output-tag set1
```

`leading_edge_plot.py` then makes charge-versus-fall-time density plots for cluster-size = 1, cluster-size > 1, and combined datasets.

```bash
python3 analysis/leading_edge_plot.py --pdata-dir <run_dir>/parquet/merged/pdata/ --tag set1
```

### 5. Cluster-size plots

`cluster_size.py` reads a leading-edge parquet dataset, computes per-event cluster size and seed charge, and saves both a seed-charge histogram split by cluster size and an overall cluster-size distribution.

It accepts either a direct parquet path or `--pdata-dir` plus `--tag`, with `set1` as the default tag.

```bash
python3 analysis/cluster_size.py <run_dir>/parquet/merged/pdata/leading_edge_all_set1.parquet
# or
python3 analysis/cluster_size.py --pdata-dir <run_dir>/parquet/merged/pdata --tag set1
```

### 6. Deposition map vs fall time

`deposition_map_vs_falltime.py` combines `DepositedCharge_merged.parquet` with the tagged leading-edge dataset, folds deposited positions into a symmetric pixel frame, and plots 2D histograms of depth versus distance from the electrode corner for fast and slow fall-time categories. Make sure the pixel size matches your geometry so the mapping is correct.

```bash
python3 analysis/deposition_map_vs_falltime.py <run_dir>/parquet/merged/DepositedCharge_merged.parquet --falltime_threshold 0.1
```
