"""Stages 5-7 - Preprocessing, modeling and evaluation.

Run:  python -m src.train

What happens, in order:
 1. Split the data: 80% training, 20% locked-away test set (stratified).
 2. Compare candidate models with 5-fold cross-validation on the TRAINING set,
    on two feature sets: raw sensors vs. raw + physics features.
 3. Compare against two baselines: "always no failure" and expert rules.
 4. Choose the alarm threshold with the lowest business cost (training data only).
 5. Train the chosen model on all training data and evaluate it ONCE on the test set.
 6. Train the failure-type model (which failure is it?).
 7. Save models, metrics and figures.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve

from src.config import (COST_FALSE_ALARM, COST_MISSED_FAILURE, MODELS_DIR,
                        RANDOM_STATE, REPORTS_DIR, ensure_dirs)
from src.data import (FEATURES_PHYSICS, FEATURES_RAW, PREDICTABLE_TYPES,
                      load_prepared, split_train_test)
from src.modeling import (best_threshold, candidate_models, cost_curve,
                          cv_splitter, out_of_fold_probabilities, ranking_scores,
                          scores_at)
from src.plotting import (FAILURE, INK_SECONDARY, MUTED, NORMAL, THIRD, save,
                          set_style)
from src.rules import FailureRules

FEATURE_SETS = {"Raw sensors": FEATURES_RAW, "Sensors + physics": FEATURES_PHYSICS}


# ---------------------------------------------------------------------------
# Step 2 and 3: compare models and baselines on the training data
# ---------------------------------------------------------------------------
def compare_models(train):
    y = train["failure"].to_numpy()
    rows, oof = [], {}
    for set_name, features in FEATURE_SETS.items():
        X = train[features]
        for model_name, model in candidate_models().items():
            if model_name.startswith("Dummy") and set_name != "Raw sensors":
                continue  # the dummy ignores features, one row is enough
            proba = out_of_fold_probabilities(model, X, y)
            threshold, cost = best_threshold(y, proba)
            row = {"model": model_name, "features": set_name,
                   **ranking_scores(y, proba),
                   "best_threshold": threshold,
                   **scores_at(y, (proba >= threshold).astype(int))}
            rows.append(row)
            oof[(model_name, set_name)] = proba
            print(f"  {model_name:30s} | {set_name:18s} | cost {row['business_cost']:4d} "
                  f"| recall {row['recall']:.2f} | PR-AUC {row['pr_auc']:.3f}")
    return pd.DataFrame(rows), oof


def cross_validate_rules(train, use_tool_wear_rule):
    """Rules also get cross-validated: thresholds learned on 4 folds, tested on the 5th."""
    y = train["failure"].to_numpy()
    pred = np.zeros(len(train), dtype=int)
    for fit_idx, check_idx in cv_splitter().split(train, y):
        rules = FailureRules(use_tool_wear_rule).fit(train.iloc[fit_idx])
        pred[check_idx] = rules.predict(train.iloc[check_idx])
    name = "Expert rules + tool-wear limit" if use_tool_wear_rule else "Expert rules (heat, power, overstrain)"
    return {"model": name, "features": "Physics thresholds", "pr_auc": np.nan,
            "roc_auc": np.nan, "best_threshold": np.nan, **scores_at(y, pred)}


# ---------------------------------------------------------------------------
# Step 6: failure-type model
# ---------------------------------------------------------------------------
def train_type_model(train, features):
    """Multi-label: one yes/no answer per failure type (24 runs have several types)."""
    failed = train[train[PREDICTABLE_TYPES].sum(axis=1) > 0]
    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=1,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(failed[features], failed[PREDICTABLE_TYPES])
    return model


def type_probabilities(model, X):
    """Probability of each failure type, as a DataFrame with one column per type."""
    raw = model.predict_proba(X)  # one array per type
    return pd.DataFrame({t: p[:, 1] for t, p in zip(PREDICTABLE_TYPES, raw)}, index=X.index)


def evaluate_type_model(model, test, features):
    failed = test[test[PREDICTABLE_TYPES].sum(axis=1) > 0]
    proba = type_probabilities(model, failed[features])
    top_type = proba.idxmax(axis=1)
    top_correct = [failed.loc[i, t] == 1 for i, t in top_type.items()]
    per_type = {}
    for t in PREDICTABLE_TYPES:
        predicted = proba[t] >= 0.5
        actual = failed[t] == 1
        per_type[t] = {
            "test_cases": int(actual.sum()),
            "recall": float((predicted & actual).sum() / max(actual.sum(), 1)),
            "precision": float((predicted & actual).sum() / max(predicted.sum(), 1)),
        }
    return {"test_failures_with_type": int(len(failed)),
            "top_type_correct": float(np.mean(top_correct)),
            "per_type": per_type}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_model_comparison(results):
    data = results.sort_values("business_cost", ascending=False)
    labels = [m if (f == "Physics thresholds" or m.startswith("Dummy")) else f"{m}  ({f})"
              for m, f in zip(data["model"], data["features"])]
    colors = [THIRD if f == "Physics thresholds" else (NORMAL if f == "Sensors + physics" else MUTED)
              for f in data["features"]]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(labels, data["business_cost"], color=colors, height=0.65)
    ax.bar_label(bars, padding=4, fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel(f"Business cost on training data, cross-validated "
                  f"(missed failure = {COST_MISSED_FAILURE}, false alarm = {COST_FALSE_ALARM})")
    ax.set_title("Lower is better: total cost of mistakes per model")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NORMAL, label="ML, sensors + physics features"),
                       Patch(color=MUTED, label="ML, raw sensors only"),
                       Patch(color=THIRD, label="Expert rules")],
              loc="upper right")
    save(fig, "model_comparison.png")


def plot_cost_curve(y, proba, threshold, name):
    thresholds, costs = cost_curve(y, proba)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(thresholds, costs, color=NORMAL)
    ax.axvline(threshold, color=FAILURE, linestyle="--", linewidth=1.5)
    ax.annotate(f"Chosen threshold {threshold:.2f}\ncost {costs[np.argmin(np.abs(thresholds - threshold))]}",
                xy=(threshold, costs.min()), xytext=(threshold + 0.08, costs.min() + (costs.max() - costs.min()) * 0.35),
                color=INK_SECONDARY, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("Alarm threshold (alarm if failure probability >= threshold)")
    ax.set_ylabel("Total business cost")
    ax.set_title(f"Choosing the alarm threshold by cost ({name}, training data)")
    save(fig, "cost_vs_threshold.png")


def plot_test_results(y_test, proba, threshold, rule_pred):
    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=NORMAL, label="Final model")
    chosen = proba >= threshold
    ax.scatter([(chosen & (y_test == 1)).sum() / (y_test == 1).sum()],
               [(chosen & (y_test == 1)).sum() / max(chosen.sum(), 1)],
               s=80, color=FAILURE, zorder=3, label=f"Chosen threshold {threshold:.2f}")
    r_tp = ((rule_pred == 1) & (y_test == 1)).sum()
    ax.scatter([r_tp / (y_test == 1).sum()], [r_tp / max(rule_pred.sum(), 1)],
               s=80, marker="s", color=THIRD, zorder=3, label="Expert rules")
    ax.axhline(y_test.mean(), color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(0.02, y_test.mean() + 0.02, "Random guessing", color=MUTED, fontsize=9)
    ax.set(xlabel="Recall (share of failures caught)",
           ylabel="Precision (share of alarms that are real)",
           title="Precision-recall on the test set", xlim=(0, 1.02), ylim=(0, 1.05))
    ax.legend(loc="center left")
    save(fig, "pr_curve_test.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= threshold).astype(int),
        display_labels=["No failure", "Failure"], cmap="Blues", colorbar=False, ax=ax)
    ax.grid(False)
    ax.set_title("Test set: what the final model gets right and wrong")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save(fig, "confusion_matrix_test.png")


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    set_style()
    df = load_prepared()
    train, test = split_train_test(df)
    y_train, y_test = train["failure"].to_numpy(), test["failure"].to_numpy()
    print(f"Training runs: {len(train)} ({y_train.sum()} failures) | "
          f"Test runs: {len(test)} ({y_test.sum()} failures)")

    print("\nCross-validated comparison on the training data:")
    results, oof = compare_models(train)
    rule_rows = [cross_validate_rules(train, False), cross_validate_rules(train, True)]
    for r in rule_rows:
        print(f"  {r['model']:50s} | cost {r['business_cost']:4d} | recall {r['recall']:.2f}")
    results = pd.concat([results, pd.DataFrame(rule_rows)], ignore_index=True)

    # Choose the best ML model by business cost (tie-break: PR-AUC)
    ml = results[~results["model"].str.startswith(("Dummy", "Expert"))]
    best = ml.sort_values(["business_cost", "pr_auc"], ascending=[True, False]).iloc[0]
    best_name, best_features_name = best["model"], best["features"]
    features = FEATURE_SETS[best_features_name]
    threshold = float(best["best_threshold"])
    print(f"\nChosen model: {best_name} with {best_features_name}, alarm threshold {threshold:.2f}")
    plot_cost_curve(y_train, oof[(best_name, best_features_name)], threshold,
                    f"{best_name}, {best_features_name.lower()}")
    plot_model_comparison(results)

    # Final model, trained on all training data, evaluated ONCE on the test set
    final_model = clone(candidate_models()[best_name]).fit(train[features], y_train)
    proba_test = final_model.predict_proba(test[features])[:, 1]
    test_scores = {**ranking_scores(y_test, proba_test),
                   **scores_at(y_test, (proba_test >= threshold).astype(int))}

    rules = FailureRules(use_tool_wear_rule=False).fit(train)
    rule_pred = rules.predict(test)
    rules_tw = FailureRules(use_tool_wear_rule=True).fit(train)
    dummy_pred = np.zeros_like(y_test)
    test_table = pd.DataFrame([
        {"model": f"{best_name} ({best_features_name})", **test_scores},
        {"model": "Expert rules (heat, power, overstrain)", **scores_at(y_test, rule_pred)},
        {"model": "Expert rules + tool-wear limit", **scores_at(y_test, rules_tw.predict(test))},
        {"model": "Dummy (always 'no failure')", **scores_at(y_test, dummy_pred),
         "accuracy": float((dummy_pred == y_test).mean())},
    ])
    print("\nTest set (used once):")
    print(test_table[["model", "recall", "precision", "caught_failures", "missed_failures",
                      "false_alarms", "business_cost"]].to_string(index=False))
    plot_test_results(y_test, proba_test, threshold, rule_pred)

    # Which failures did the final model miss?
    missed = test[(y_test == 1) & (proba_test < threshold)]
    missed_types = {t: int(missed[t].sum()) for t in PREDICTABLE_TYPES + ["RNF"]}
    missed_types["no type given"] = int((missed[PREDICTABLE_TYPES + ["RNF"]].sum(axis=1) == 0).sum())

    # Failure-type model
    type_model = train_type_model(train, features)
    type_scores = evaluate_type_model(type_model, test, features)
    print(f"\nFailure-type model: top predicted type correct for "
          f"{type_scores['top_type_correct']:.0%} of {type_scores['test_failures_with_type']} test failures")

    # Save everything
    joblib.dump({"model": final_model, "features": features, "threshold": threshold,
                 "model_name": best_name, "feature_set": best_features_name},
                MODELS_DIR / "failure_model.joblib")
    joblib.dump({"model": type_model, "features": features, "types": PREDICTABLE_TYPES},
                MODELS_DIR / "type_model.joblib")
    joblib.dump(rules, MODELS_DIR / "expert_rules.joblib")

    results.round(4).to_csv(REPORTS_DIR / "model_comparison_cv.csv", index=False)
    test_table.round(4).to_csv(REPORTS_DIR / "test_results.csv", index=False)
    metrics = {
        "train_runs": int(len(train)), "train_failures": int(y_train.sum()),
        "test_runs": int(len(test)), "test_failures": int(y_test.sum()),
        "costs": {"missed_failure": COST_MISSED_FAILURE, "false_alarm": COST_FALSE_ALARM},
        "chosen_model": best_name, "chosen_features": best_features_name,
        "alarm_threshold": threshold,
        "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in test_scores.items()},
        "test_missed_failures_by_type": missed_types,
        "rules_thresholds": {k: ({kk: round(float(vv), 1) for kk, vv in v.items()}
                                 if isinstance(v, dict) else round(float(v), 2))
                             for k, v in rules.thresholds_.items()},
        "type_model": type_scores,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved models to {MODELS_DIR} and metrics to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
