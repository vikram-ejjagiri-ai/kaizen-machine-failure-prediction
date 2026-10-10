"""Stage 8 - Explainability: why does the model raise an alarm?

Run:  python -m src.explain   (after python -m src.train)

SHAP splits every prediction into contributions per feature:
    final prediction = average prediction + contribution(feature 1) + contribution(feature 2) + ...
A positive contribution pushes towards "failure", a negative one towards "normal".
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, REPORTS_DIR, ensure_dirs
from src.data import load_prepared, split_train_test
from src.eda import LABELS
from src.plotting import FAILURE, INK_SECONDARY, NORMAL, save, set_style

FEATURE_LABELS = {**LABELS, "product_quality": "Product quality (L=0, M=1, H=2)"}


def failure_shap_values(model, X):
    """SHAP values for the 'failure' class, shape (rows, features)."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):          # older shap: one array per class
        values = values[1]
    elif values.ndim == 3:                # newer shap: (rows, features, classes)
        values = values[:, :, 1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    return values, base


def plot_global(values, X):
    importance = pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh([FEATURE_LABELS[c] for c in importance.index], importance.values,
                   color=NORMAL, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Average impact on predicted failure probability (mean |SHAP|)")
    ax.set_title("What drives the model's predictions overall (test set)")
    ax.grid(axis="y", visible=False)
    save(fig, "shap_global_importance.png")
    return importance.sort_values(ascending=False)


def plot_beeswarm(values, X):
    shap.summary_plot(values, X.rename(columns=FEATURE_LABELS), show=False, plot_size=(10, 6))
    fig = plt.gcf()
    plt.title("How each feature pushes towards failure (red = high value, blue = low value)",
              fontsize=11, fontweight="bold")
    save(fig, "shap_beeswarm.png")


def plot_single(values, base, X, row_position, proba, title, filename):
    """Contribution of each feature to ONE prediction, as a horizontal bar chart."""
    contrib = pd.Series(values[row_position], index=X.columns)
    order = contrib.abs().sort_values().index
    row = X.iloc[row_position]
    labels = [f"{FEATURE_LABELS[c]} = {row[c]:.1f}" for c in order]
    colors = [FAILURE if contrib[c] > 0 else NORMAL for c in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, contrib[order], color=colors, height=0.6)
    ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xlabel("Contribution to failure probability "
                  "(orange = towards failure, blue = towards normal)")
    ax.set_title(f"{title}\nBaseline {base:.2f}  ->  predicted failure probability {proba:.2f}",
                 fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, filename)
    return {FEATURE_LABELS[c]: round(float(contrib[c]), 3) for c in order[::-1]}


def main():
    ensure_dirs()
    set_style()
    bundle = joblib.load(MODELS_DIR / "failure_model.joblib")
    model, features, threshold = bundle["model"], bundle["features"], bundle["threshold"]

    _, test = split_train_test(load_prepared())
    X = test[features]
    proba = model.predict_proba(X)[:, 1]
    values, base = failure_shap_values(model, X)

    importance = plot_global(values, X)
    plot_beeswarm(values, X)

    y = test["failure"].to_numpy()
    positions = np.arange(len(test))
    # Example 1: a correctly raised alarm (the most confident overstrain case)
    caught_osf = positions[(y == 1) & (proba >= threshold) & (test["OSF"].to_numpy() == 1)]
    alarm_pos = caught_osf[np.argmax(proba[caught_osf])]
    alarm = plot_single(values, base, X, alarm_pos, proba[alarm_pos],
                        f"Why the model raised this alarm (run UDI {test.iloc[alarm_pos]['UDI']}, "
                        "actual: overstrain failure)", "shap_example_alarm.png")
    # Example 2: a missed tool-wear failure
    missed_twf = positions[(y == 1) & (proba < threshold) & (test["TWF"].to_numpy() == 1)]
    missed_pos = missed_twf[np.argmax(test.iloc[missed_twf]["tool_wear_min"].to_numpy())]
    missed = plot_single(values, base, X, missed_pos, proba[missed_pos],
                         f"Why the model missed this failure (run UDI {test.iloc[missed_pos]['UDI']}, "
                         "actual: tool wear failure)", "shap_example_missed.png")

    summary = {
        "global_importance_mean_abs_shap": importance.round(4).to_dict(),
        "example_alarm": {"udi": int(test.iloc[alarm_pos]["UDI"]),
                          "probability": round(float(proba[alarm_pos]), 3),
                          "contributions": alarm},
        "example_missed": {"udi": int(test.iloc[missed_pos]["UDI"]),
                           "probability": round(float(proba[missed_pos]), 3),
                           "contributions": missed},
    }
    (REPORTS_DIR / "explainability.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
