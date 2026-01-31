#!/usr/bin/env python3
"""Paper-style plots from reconstructed ER-only results.

Input: CSV with columns:
family,n,seed,p,opt,base,tr,warm,randFT,gain_warm_base,rho,grad,hit,gate,reason,donor,scale,time_s

Usage:
  python paper_artifacts/plot_paper_figs_from_csv.py --csv paper_artifacts/er_clean_results.csv --out paper_artifacts/figures
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def heatmap(table, title, out_path, fmt="{:.2f}", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    arr = table.values.astype(float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("p")
    ax.set_ylabel("n")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([str(i) for i in table.index])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            s = "—" if np.isnan(val) else fmt.format(val)
            ax.text(j, i, s, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figs")
    ap.add_argument(
        "--paper_title", default="ER_dense_p05_weighted — Surrogate Transfer Summary"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    rename_map = {
        "gain_warm_minus_base": "gain_warm_base",
        "fp_rho": "rho",
        "fp_grad": "grad",
        "fp_hit": "hit",
        "skip_reason": "reason",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    num_cols = [
        "opt",
        "base",
        "tr",
        "warm",
        "randFT",
        "gain_warm_base",
        "rho",
        "grad",
        "hit",
        "scale",
        "time_s",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].map(_safe_float)

    df["is_pass"] = df.get("gate", "") == "PASS"
    df["has_gain"] = np.isfinite(df.get("gain_warm_base", np.nan))

    ensure_dir(args.out)

    ns = sorted(df["n"].unique().tolist())
    ps = sorted(df["p"].unique().tolist())

    pass_rate = (
        df.groupby(["n", "p"])["is_pass"]
        .mean()
        .unstack("p")
        .reindex(index=ns, columns=ps)
    )
    heatmap(
        pass_rate,
        f"{args.paper_title}\nPass rate (gate==PASS)",
        os.path.join(args.out, "heatmap_pass_rate.png"),
        fmt="{:.2f}",
        vmin=0.0,
        vmax=1.0,
    )

    df_gain = df[df["has_gain"]].copy()
    mean_gain = (
        df_gain.groupby(["n", "p"])["gain_warm_base"]
        .mean()
        .unstack("p")
        .reindex(index=ns, columns=ps)
    )
    med_gain = (
        df_gain.groupby(["n", "p"])["gain_warm_base"]
        .median()
        .unstack("p")
        .reindex(index=ns, columns=ps)
    )

    heatmap(
        mean_gain,
        f"{args.paper_title}\nMean gain (warm - base) over finite entries",
        os.path.join(args.out, "heatmap_mean_gain.png"),
    )
    heatmap(
        med_gain,
        f"{args.paper_title}\nMedian gain (warm - base) over finite entries",
        os.path.join(args.out, "heatmap_median_gain.png"),
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    box_data = []
    labels = []
    for p in ps:
        g = df_gain[df_gain["p"] == p]["gain_warm_base"].values
        box_data.append(g)
        labels.append(str(p))
    ax.boxplot(box_data, labels=labels, showfliers=True)
    ax.set_title(f"{args.paper_title}\nGain distribution (warm - base) by p")
    ax.set_xlabel("p")
    ax.set_ylabel("gain (warm - base)")
    ax.axhline(0.0, linewidth=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "box_gain_by_p.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tmp = df_gain[np.isfinite(df_gain["rho"])].copy()
    ax.scatter(tmp["rho"].values, tmp["gain_warm_base"].values, alpha=0.8)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(f"{args.paper_title}\nFingerprint rho vs gain")
    ax.set_xlabel("rho")
    ax.set_ylabel("gain (warm - base)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scatter_fp_rho_vs_gain.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    tmp = df_gain[np.isfinite(df_gain["grad"])].copy()
    ax.scatter(tmp["grad"].values, tmp["gain_warm_base"].values, alpha=0.8)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(f"{args.paper_title}\nFingerprint grad vs gain")
    ax.set_xlabel("grad")
    ax.set_ylabel("gain (warm - base)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scatter_fp_grad_vs_gain.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    donor_counts = df[df["is_pass"]].get("donor", pd.Series([], dtype=str)).value_counts()
    if len(donor_counts) == 0:
        donor_counts = df.get("donor", pd.Series([], dtype=str)).value_counts()
    ax.bar(donor_counts.index.astype(str).tolist(), donor_counts.values.tolist())
    ax.set_title(f"{args.paper_title}\nDonor family counts (PASS only)")
    ax.set_xlabel("donor")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "donor_counts.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    skip = df[df.get("gate", "") != "PASS"].copy()
    if "reason" in skip.columns:
        reasons = skip["reason"].fillna("unknown").astype(str).value_counts()
    else:
        reasons = pd.Series(dtype=int)
    if len(reasons) == 0:
        reasons = pd.Series({"none": 0})
    ax.bar(range(len(reasons)), reasons.values.tolist())
    ax.set_xticks(range(len(reasons)))
    ax.set_xticklabels(reasons.index.astype(str).tolist(), rotation=25, ha="right")
    ax.set_title(f"{args.paper_title}\nSkip reasons (gate != PASS)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "skip_reasons.png"), dpi=200)
    plt.close(fig)

    lines = []
    lines.append("=== ER-only reconstructed summary ===")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Unique (n,p): {df[['n','p']].drop_duplicates().shape[0]}")
    lines.append(f"PASS rate overall: {df['is_pass'].mean():.3f}")
    if len(df_gain) > 0:
        lines.append(f"Mean gain over finite entries: {df_gain['gain_warm_base'].mean():.3f}")
        lines.append(
            f"Median gain over finite entries: {df_gain['gain_warm_base'].median():.3f}"
        )
        best = df_gain.sort_values("gain_warm_base", ascending=False).head(10)
        lines.append("\nTop-10 gains (warm-base):")
        lines.append(
            best[["n", "p", "seed", "gain_warm_base", "base", "warm", "donor", "gate"]].to_string(
                index=False
            )
        )
    with open(os.path.join(args.out, "paper_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved figures to:", args.out)


if __name__ == "__main__":
    main()
