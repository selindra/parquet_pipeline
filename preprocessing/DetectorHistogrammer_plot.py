#!/usr/bin/env python3
"""
Extract threshold + mean efficiency (Mean Y) + cluster size mean from ROOT files,
then make:
1) efficiency + cluster size plot

Usage:
    python DetectorHistogrammer_plot.py --folder /path/to/root_folder \
                                              --eff-file Efficiency_vs_threshold_Vbb_4.8_AO10P.txt

    python3 DetectorHistogrammer_plot.py --folder $DIR/output/trycor_efficiency_4 --scale e --capcitance-fF 2.8                                          
"""

import sys
import os
import math
import re
import argparse

import uproot
import numpy as np
import matplotlib.pyplot as plt

# ROOT histogram paths (APTS OPAMP setup)
EFF_PATH = "DetectorHistogrammer/APTS_OPAMP/efficiency/efficiency_vs_x;1"
CLUSTER_SIZE_PATH = "DetectorHistogrammer/APTS_OPAMP/cluster_size/cluster_size_x;1"

# Scaling options: choose via command line
#   - "e"     : convert DAC steps to electrons using fixed capacitance
#   - "emv"   : convert to mV using fixed e/mV (e.g., 18.51 e/mV)
#   - "mv"    : keep x‑axis as DAC steps (label as mV for convenience)
VALID_SCALE = ("e", "emv", "mv")

# --------------- Configuration / parameters (documented) --------------------

# Default scaling parameter (can be overridden by command line)
DEFAULT_SCALE = "emv"  # convert -threshold * emv -> mV

# Charge / gain factors
EMV = 18.52
CAPACITANCE_fF = 2.71

# Default experiment file name
DEFAULT_EXP_EFF_NAME = "Efficiency_vs_threshold_Vbb_4.8_AO10P.txt"
DEFAULT_EXP_RES_NAME = "Spatial_resolution_vbb_4.8.txt"


def parse_args():
    """Parse command‑line arguments and perform basic checks."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=".",
        help="Folder containing .root files (default: current directory)."
    )
    parser.add_argument(
        "--eff-file",
        type=str,
        default=None,
        help="Experimental efficiency file path (default: {} in folder).".format(DEFAULT_EXP_EFF_NAME),
    )
    parser.add_argument(
        "--res-file",
        type=str,
        default=None,
        help="Experimental cluster-size file path (default: {} in folder).".format(DEFAULT_EXP_RES_NAME),
    )

    parser.add_argument(
        "--scale",
        choices=VALID_SCALE,
        default=DEFAULT_SCALE,
        help=f"Scaling of x‑axis: {VALID_SCALE} (default: {DEFAULT_SCALE})."
    )
    parser.add_argument(
        "--emv",
        type=float,
        default=EMV,
        help=f"Calibration factor from mV to e used when --scale is 'emv', (default: {EMV})."
    )
    parser.add_argument(
        "--capcitance-fF",
        type=float,
        default=CAPACITANCE_fF,
        help=f"Effective capacitance in fF used when --scale is 'e' (default: {CAPACITANCE_fF})."
    )

    return parser.parse_args()


def extract_threshold(filename):
    """
    Extract threshold value from ROOT filename pattern like "thr-<value>.root".

    Returns:
        float (threshold) or None if not found.
    """
    match = re.search(r"thr-(\d+(?:\.\d+)?)", filename)
    return float(match.group(1)) if match else None

def load_experimental_cluster_data(folder, res_file_option):
    """
    Load experimental cluster size data.

    Args:
        folder: Parent folder.
        res_file_option: Path given by user or None.

    Returns:
        dict with keys for cluster size.
    """
    res_name = res_file_option or DEFAULT_EXP_RES_NAME
    exp_res_file = os.path.join(folder, res_name)
    if not os.path.exists(exp_res_file):
        print(f"Experimental cluster-size file not found: {exp_res_file}")
        sys.exit(1)

    data = np.loadtxt(exp_res_file)

    return {
        "cluster_th": data[:, 2],
        "cluster_size": data[:, 5],
        "cluster_size_err": data[:, 6],
    }



def get_profile_mean_and_err(root_file_path, hist_path):
    """
    Extract mean Y and error from a TProfile stored at `hist_path`.

    Args:
        root_file_path: Path to the .root file.
        hist_path: Key in the ROOT file.

    Returns:
        (mean_y, err_y) or (None, None) if no entries.
    """
    with uproot.open(root_file_path) as f:
        h = f[hist_path]
        entries = float(h.member("fTsumw"))
        sum_y = float(h.member("fTsumwy"))
        sum_y2 = float(h.member("fTsumwy2"))
        if entries <= 0:
            return None, None
        mean_y = sum_y / entries
        var_y = (sum_y2 / entries) - mean_y**2
        stdev_y = math.sqrt(max(var_y, 0.0))
        err = stdev_y / math.sqrt(entries)
        return mean_y, err


def get_th1_stats(root_file_path, hist_path):
    """
    Extract mean and standard deviation from a 1D TH1F stored at `hist_path`.

    Args:
        root_file_path: Path to the .root file.
        hist_path: Key in the ROOT file.

    Returns:
        (mean, std) or (None, None) if no entries.
    """
    with uproot.open(root_file_path) as f:
        h = f[hist_path]
        values = h.values(flow=False)
        edges = h.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = np.sum(values)
        if total <= 0:
            return None, None
        mean = np.sum(values * centers) / total
        var = np.sum(values * (centers - mean)**2) / total
        stdev = math.sqrt(max(var, 0.0))
        return float(mean), float(stdev)


def scale_x_axis(thresholds, scale, emv, capacitance_fF):
    """
    Convert threshold (DAC steps) to desired x‑axis quantity.

    Args:
        thresholds: Array of raw thresholds (positive DAC steps).
        scale: One of "e", "emv", "mv".
        emv: e/mV factor.
        capacitance_fF: Detector capacitance in fF.

    Returns:
        xvals: Array of scaled x values (electrons or mV).
    """
    steps = -thresholds
    if scale == "e":
        xvals = steps * capacitance_fF * 10 / 1.602
    elif scale == "emv":
        xvals = steps * emv
    else:  # "mv"
        xvals = steps
    return xvals


def load_experimental_eff_data(folder, eff_file_option):
    """
    Load experimental efficiency data (threshold, eff, errors).

    Args:
        folder: Parent folder.
        eff_file_option: Path given by user or None.

    Returns:
        dict with keys: "threshold", "eff", "err_up", "err_down".
    """
    eff_name = eff_file_option or DEFAULT_EXP_EFF_NAME
    exp_eff_file = os.path.join(folder, eff_name)
    if not os.path.exists(exp_eff_file):
        print(f"Experimental efficiency file not found: {exp_eff_file}")
        sys.exit(1)

    data = np.loadtxt(exp_eff_file)
    return {
        "threshold": data[:, 1],
        "eff": data[:, 2] * 100,
        "err_up": data[:, 3] * 100,
        "err_down": data[:, 4] * 100,
    }


def main():
    args = parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Folder does not exist: {folder}")
        sys.exit(1)

    root_files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".root") and extract_threshold(f) is not None
    )
    if not root_files:
        print("No ROOT files matching 'thr-<value>' were found.")
        sys.exit(1)

    records = []
    for fname in root_files:
        thr = extract_threshold(fname)
        fpath = os.path.join(folder, fname)

        eff_mean, eff_err = get_profile_mean_and_err(fpath, EFF_PATH)
        if eff_mean is None:
            continue

        cl_mean, cl_std = get_th1_stats(fpath, CLUSTER_SIZE_PATH)

        records.append((
            thr,
            eff_mean * 100,
            eff_err * 100,
            cl_mean,
            cl_std,
        ))

    records.sort(key=lambda x: x[0])
    if not records:
        print("No valid records found in the input files.")
        sys.exit(1)

    thresholds   = np.array([r[0] for r in records], dtype=float)
    efficiency   = np.array([r[1] for r in records], dtype=float)
    err_sim      = np.array([r[2] for r in records], dtype=float)
    cluster_size = np.array([r[3] for r in records], dtype=float)
    cluster_err  = np.array([r[4] for r in records], dtype=float)

    out_txt = os.path.join(folder, "extracted_threshold_scan_results.txt")
    with open(out_txt, "w") as fout:
        fout.write(
            "threshold,efficiency_percent,efficiency_err_percent,cluster_size_mean,cluster_size_std\n"
        )
        for r in records:
            vals = [str(v) for v in r]
            fout.write(",".join(vals) + "\n")

    EMV = args.emv
    CAPACITANCE_fF = args.capcitance_fF
    xvals = scale_x_axis(thresholds, args.scale, EMV, CAPACITANCE_fF)

    exp_eff = load_experimental_eff_data(folder, args.eff_file)
    exp_cluster = load_experimental_cluster_data(folder, args.res_file)


    # ---------- Efficiency figure ----------
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.errorbar(
        xvals[1:-1] * (-1),
        efficiency[1:-1],
        yerr=err_sim[1:-1],
        fmt="o-",
        color="#56B4E9",
        capsize=4,
        label="Simulation efficiency",
    )

    if args.scale == "mv":
        exp_x_eff = None
    else:
        exp_x_eff = exp_eff["threshold"] if args.scale == "e" else exp_eff["threshold"]

    if exp_x_eff is not None:
        ax1.errorbar(
            exp_x_eff,
            exp_eff["eff"],
            yerr=[exp_eff["err_down"], exp_eff["err_up"]],
            fmt="s--",
            color="#F98128",
            capsize=4,
            label="Experimental efficiency",
        )

    ax1.set_ylabel("Detection efficiency (%), Vsub = -4.8V", fontsize=15)
    ax1.tick_params(axis="both", labelsize=15)
    ax1.grid(True)
    ax1.set_ylim(80, 101)
    ax1.axhline(99, linestyle="--", linewidth=1, color="silver")
    ax1.text(-0.05, 98.5, "99", transform=ax1.get_yaxis_transform(), va="bottom")
    ax1.set_xlabel("Threshold (e)" if args.scale != "mv" else "Threshold (mV)", fontsize=15)
    ax1.legend(fontsize=15, loc="best")

    plt.tight_layout()

    if args.scale == "e":
        out_eff = os.path.join(folder, f"efficiency_{args.scale}_capacitance{CAPACITANCE_fF}.png")
    elif args.scale == "emv":
        out_eff = os.path.join(folder, f"efficiency_{args.scale}_emv{EMV}.png")
    else:
        out_eff = os.path.join(folder, f"efficiency_{args.scale}.png")

    plt.savefig(out_eff, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ---------- Cluster size figure ----------
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.errorbar(
        xvals[2:-2] * (-1),
        cluster_size[2:-2],
        yerr=cluster_err[2:-2],
        fmt="d-",
        color="#009E73",
        ecolor="#009E7493",
        elinewidth=1,
        capsize=4,
        label="Simulation cluster size",
    )

    if args.scale == "mv":
        exp_x_cl = None
    else:
        exp_x_cl = exp_cluster["cluster_th"] if args.scale == "e" else exp_cluster["cluster_th"]

    if exp_x_cl is not None:
        ax2.errorbar(
            exp_x_cl[1:],
            exp_cluster["cluster_size"][1:],
            # yerr=exp_cluster["cluster_size_err"][2:],
            fmt="^--",
            color="#F98128",
            capsize=4,
            label="Experimental cluster size",
        )

    ax2.set_ylabel("Cluster size, Vsub = -4.8V", fontsize=15)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.grid(True)
    ax2.set_xlabel("Threshold (e)" if args.scale != "mv" else "Threshold (mV)", fontsize=15)
    ax2.legend(fontsize=15, loc="best")
    plt.tight_layout()

    if args.scale == "e":
        out_cl = os.path.join(folder, f"cluster_size_{args.scale}_capacitance{CAPACITANCE_fF}.png")
    elif args.scale == "emv":
        out_cl = os.path.join(folder, f"cluster_size_{args.scale}_emv{EMV}.png")
    else:
        out_cl = os.path.join(folder, f"cluster_size_{args.scale}.png")

    plt.savefig(out_cl, dpi=300, bbox_inches="tight")
    plt.close(fig2)


    print(f"Saved extracted data to: {out_txt}")
    print(f"Saved plot to: {out_cl}")


if __name__ == "__main__":
    main()
