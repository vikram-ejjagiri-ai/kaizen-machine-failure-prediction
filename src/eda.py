"""Stage 4 - Exploratory Data Analysis.

Run:  python -m src.eda

Important habit: the EDA looks at the TRAINING data only. If we studied the
test set too, our ideas (and the rules we derive) would be shaped by data that
is supposed to be unseen - a subtle form of cheating.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.config import REPORTS_DIR, ensure_dirs
from src.data import (PHYSICS_FEATURES, PREDICTABLE_TYPES, SENSORS,
                      load_prepared, split_train_test)
from src.plotting import (DIVERGING, FAILURE, INK_SECONDARY, MUTED, NEUTRAL,
                          NORMAL, save, set_style)

LABELS = {
    "air_temp_k": "Air temperature [K]",
    "process_temp_k": "Process temperature [K]",
    "speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "temp_diff_k": "Temperature difference [K]",
    "power_w": "Mechanical power [W]",
    "strain_nm_min": "Strain = tool wear x torque",
}
TYPE_NAMES = {"TWF": "Tool wear", "HDF": "Heat dissipation",
              "PWF": "Power", "OSF": "Overstrain"}


def plot_distributions(train):
    features = SENSORS + PHYSICS_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flat, features):
        bins = np.histogram_bin_edges(train[col], bins=40)
        for value, color, label in [(0, NORMAL, "Normal run"), (1, FAILURE, "Failure")]:
            ax.hist(train.loc[train["failure"] == value, col], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=label)
        ax.set_title(LABELS[col], fontsize=10)
        ax.set_yticks([])
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Failed runs vs normal runs (each group scaled to the same height)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "distributions_by_failure.png")


def plot_correlations(train):
    cols = SENSORS + PHYSICS_FEATURES + ["failure"]
    corr = train[cols].corr()
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    names = [LABELS.get(c, "Failure") for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation between features (training data)")
    save(fig, "correlation_heatmap.png")
    return corr["failure"].drop("failure")


def plot_correlation_blind_spot(train):
    """Power barely correlates with failure - yet failures cluster at BOTH extremes."""
    bands = pd.cut(train["power_w"], bins=np.arange(1000, 11001, 500))
    rate = train.groupby(bands, observed=True)["failure"].agg(["mean", "size"])
    rate = rate[rate["size"] >= 10]  # skip almost-empty bands
    centers = [b.mid for b in rate.index]
    corr = train["power_w"].corr(train["failure"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(centers, rate["mean"] * 100, width=420, color=NORMAL)
    ax.set_xlabel("Mechanical power [W] (500 W bands, only bands with 10+ runs)")
    ax.set_ylabel("Failure rate [%]")
    ax.set_title(f"Correlation of power with failure is only {corr:.2f}, "
                 "yet failures pile up at both ends")
    save(fig, "correlation_blind_spot_power.png")
    return float(corr)


def plot_failure_mechanisms(train):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    rest = dict(s=6, color=NEUTRAL, alpha=0.5, label="All other runs")
    hit = dict(s=22, color=FAILURE, edgecolor="white", linewidth=0.6)

    ax = axes[0, 0]
    other = train[train["HDF"] == 0]
    ax.scatter(other["temp_diff_k"], other["speed_rpm"], **rest)
    h = train[train["HDF"] == 1]
    ax.scatter(h["temp_diff_k"], h["speed_rpm"], label="Heat dissipation failure", **hit)
    ax.set(xlabel=LABELS["temp_diff_k"], ylabel=LABELS["speed_rpm"],
           title="Heat dissipation: small temperature difference AND low speed")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    other = train[train["PWF"] == 0]
    ax.scatter(other["speed_rpm"], other["torque_nm"], **rest)
    p = train[train["PWF"] == 1]
    ax.scatter(p["speed_rpm"], p["torque_nm"], label="Power failure", **hit)
    ax.set(xlabel=LABELS["speed_rpm"], ylabel=LABELS["torque_nm"],
           title="Power: failures at both ends of the power range")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    other = train[train["OSF"] == 0]
    ax.scatter(other["tool_wear_min"], other["torque_nm"], **rest)
    o = train[train["OSF"] == 1]
    ax.scatter(o["tool_wear_min"], o["torque_nm"], label="Overstrain failure", **hit)
    ax.set(xlabel=LABELS["tool_wear_min"], ylabel=LABELS["torque_nm"],
           title="Overstrain: worn tool under high torque")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    bands = pd.cut(train["tool_wear_min"], bins=np.arange(0, 261, 20), right=False)
    rate = train.groupby(bands, observed=True)["TWF"].mean() * 100
    ax.bar([b.mid for b in rate.index], rate.values, width=16, color=NORMAL)
    ax.set(xlabel=LABELS["tool_wear_min"] + " (20-minute bands)",
           ylabel="Tool wear failure rate [%]",
           title="Tool wear: failures only appear after ~200 minutes")

    fig.suptitle("Four failure mechanisms, rediscovered from the data",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "failure_mechanisms.png")


def plot_variant_breakdown(train):
    """Why do low-quality (L) products fail more? Split the failure rate by type."""
    order = ["L", "M", "H"]
    panels = ["failure"] + PREDICTABLE_TYPES
    titles = ["All failures"] + [TYPE_NAMES[t] for t in PREDICTABLE_TYPES]
    rates = train.groupby("product_type")[panels].mean().loc[order] * 100

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    for ax, col, title in zip(axes, panels, titles):
        bars = ax.bar(order, rates[col], color=NORMAL, width=0.6)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, color=INK_SECONDARY)
        ax.set_title(title)
    axes[0].set_ylabel("Failure rate [%]")
    fig.suptitle("Failure rate per product variant: the biggest gap is overstrain",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "variant_breakdown.png")
    return rates.round(2)


def summarize(train):
    """Numbers quoted in docs/04_eda_findings.md."""
    def describe(mask, cols):
        return train.loc[mask, cols].agg(["min", "max"]).round(1).to_dict()

    return {
        "train_rows": int(len(train)),
        "train_failures": int(train["failure"].sum()),
        "hdf": describe(train["HDF"] == 1, ["temp_diff_k", "speed_rpm"]),
        "pwf_power_w": describe(train["PWF"] == 1, ["power_w"]),
        "osf_min_strain_by_variant": train[train["OSF"] == 1]
        .groupby("product_type")["strain_nm_min"].min().round(0).to_dict(),
        "twf_tool_wear": describe(train["TWF"] == 1, ["tool_wear_min"]),
        "median_by_failure": train.groupby("failure")[SENSORS + PHYSICS_FEATURES]
        .median().round(1).to_dict(orient="index"),
    }


def main():
    ensure_dirs()
    set_style()
    df = load_prepared()
    train, _ = split_train_test(df)

    plot_distributions(train)
    corr_with_failure = plot_correlations(train)
    power_corr = plot_correlation_blind_spot(train)
    plot_failure_mechanisms(train)
    variant_rates = plot_variant_breakdown(train)

    summary = summarize(train)
    summary["correlation_with_failure"] = corr_with_failure.round(3).to_dict()
    summary["power_correlation"] = round(power_corr, 3)
    summary["failure_rate_by_variant_pct"] = variant_rates.to_dict(orient="index")
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
