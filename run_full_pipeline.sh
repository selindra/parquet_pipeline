#!/bin/bash
DIR=""

#####Preprocessing#####
python3 preprocessing/extract_root.py $DIR/Tree.root APTS_OPAMP --multiple_files 
python3 preprocessing/merge_parquet.py $DIR/parquet/
python3 preprocessing/extract_seed_pixel_data.py --input $DIR/parquet/merged/PixelPulse_merged.parquet

#####Data inspection#####
python3 analysis/pixel_amplitude_hist.py $DIR/parquet/merged/PixelPulse_merged.parquet --input-type pulse
python3 analysis/pixel_amplitude_hist.py $DIR/parquet/merged/PixelCharge_merged.parquet --input-type charge
python3 analysis/inspect_parquet.py $DIR/parquet/merged/PixelPulse_merged.parquet
python3 analysis/plot_waveforms_range.py --file $DIR/parquet/merged/pdata/cluster_size1.parquet

#####Analysis#####
python3 analysis/calibration.py --pdata-dir $DIR/parquet/merged/pdata/
python3 analysis/leading_edge_extract.py --pdata-dir $DIR/parquet/merged/pdata/ 
python3 analysis/leading_edge_plot.py --pdata-dir $DIR/parquet/merged/pdata/
python3 analysis/cluster_size.py $DIR/parquet/merged/pdata/leading_edge_all_set1.parquet
python3 analysis/deposition_map_vs_falltime.py $DIR/parquet/merged/DepositedCharge_merged.parquet
