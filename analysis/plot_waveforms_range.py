import argparse
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os

"""
plot_waveforms_range.py

Read a Parquet file containing waveform data and plot up to MAX_WAVEFORMS
waveforms whose amplitude, expressed in electrons, falls within a specified
range (AMP_MIN to AMP_MAX).

Amplitude is computed using a conversion factor from mV to electrons.
The plot is saved as a PNG in the same directory as the input Parquet file.

Usage:
    python3 plot_waveforms_range.py --file <path_to_parquet_file>

Example:
    python3 plot_waveforms_range.py --file $DIR/parquet/merged/pdata/cluster_size1.parquet
"""



# --------------------------------------------------
# SETTINGS (in ELECTRONS)
# --------------------------------------------------
AMP_MIN = 250
AMP_MAX = 600
CONVERSION_FACTOR = 18.52  # e / mV
MAX_WAVEFORMS = 3


# --------------------------------------------------
# LOAD AND FILTER WAVEFORMS
# --------------------------------------------------
def load_waveforms(parquet_file):

    print("Reading:", parquet_file)

    df = pq.read_table(parquet_file).to_pandas()

    waveform_col = None
    for c in df.columns:
        if isinstance(df[c].iloc[0], (list, np.ndarray)):
            waveform_col = c
            break

    print("Waveform column:", waveform_col)

    selected_waveforms = []
    count_total = 0

    for _, row in df.iterrows():

        wf = row[waveform_col]

        if not isinstance(wf, (list, np.ndarray)):
            continue

        wf = np.asarray(wf)

        baseline = np.median(wf[:10])
        amp_mV = baseline - np.min(wf)
        amp_e  = amp_mV * CONVERSION_FACTOR

        count_total += 1

        if AMP_MIN <= amp_e <= AMP_MAX:
            selected_waveforms.append(wf)

    print(f"Total waveforms checked: {count_total}")
    print(f"Selected waveforms in range {AMP_MIN}-{AMP_MAX} e: {len(selected_waveforms)}")

    return selected_waveforms


# --------------------------------------------------
# PLOT
# --------------------------------------------------
def plot_waveforms(waveforms, parquet_file):

    if len(waveforms) == 0:
        print("No waveforms to plot")
        return

    out_dir = os.path.join(os.path.dirname(parquet_file), "plots")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(
        out_dir,
        f"waveforms_{AMP_MIN}_{AMP_MAX}_e.png"
    )

    plt.figure(figsize=(8,6))

    count = 0

    for wf in waveforms:

        if count >= MAX_WAVEFORMS:
            break

        wf = np.asarray(wf)

        baseline = np.median(wf[:10])
        wf_shifted = wf - baseline

        plt.plot(wf_shifted, alpha=0.9)

        count += 1

    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (baseline-subtracted mV)")
    plt.title(f"Waveforms {AMP_MIN}-{AMP_MAX} e (N={count})")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_file, dpi=150)
    plt.close()

    print("Saved:", out_file)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)

    args = parser.parse_args()

    waveforms = load_waveforms(args.file)

    plot_waveforms(waveforms, args.file)