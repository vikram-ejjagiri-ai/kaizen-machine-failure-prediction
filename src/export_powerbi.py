"""Stage 9 - Data for the Power BI dashboard.

Run:  python -m src.export_powerbi   (after python -m src.train)

Every one of the 10,000 runs gets a risk score from a model that did NOT see
that run during training (5-fold cross-scoring). That way the dashboard shows
honest performance, not the model's memory of its own training data.

Output files (in the powerbi/ folder):
  machine_runs_scored.csv   one row per production run - the main table
  model_results.csv         test-set results: final model vs. baselines
  feature_importance.csv    what drives the predictions (from SHAP)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import MODELS_DIR, POWERBI_DIR, REPORTS_DIR, ensure_dirs
from src.data import FAILURE_TYPES, PREDICTABLE_TYPES, load_prepared
from src.modeling import candidate_models, cv_splitter
from src.rules import FailureRules
from src.train import train_type_model, type_probabilities

TYPE_NAMES = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power",
              "OSF": "Overstrain", "RNF": "Random"}
REASON_TEXT = {"HDF": "Poor heat dissipation (low temp. difference + low speed)",
               "PWF": "Power outside safe range",
               "OSF": "Tool overstrained (wear x torque above limit)",
               "TWF": "Tool wear above limit"}


def actual_type_text(row):
    present = [TYPE_NAMES[t] for t in FAILURE_TYPES if row[t] == 1]
    if row["failure"] == 1 and not present:
        return "Unknown"
    return " + ".join(present) if present else "No failure"


def cross_score(df, model_name, features):
    """Risk score, predicted type and rule reason for every run, 5-fold."""
    y = df["failure"].to_numpy()
    risk = np.zeros(len(df))
    type_proba = pd.DataFrame(0.0, index=df.index, columns=PREDICTABLE_TYPES)
    reasons = pd.DataFrame(False, index=df.index, columns=["HDF", "PWF", "OSF", "TWF"])
    for fit_idx, score_idx in cv_splitter().split(df, y):
        fit, score = df.iloc[fit_idx], df.iloc[score_idx]
        model = clone(candidate_models()[model_name]).fit(fit[features], fit["failure"])
        risk[score_idx] = model.predict_proba(score[features])[:, 1]
        type_model = train_type_model(fit, features)
        type_proba.iloc[score_idx] = type_probabilities(type_model, score[features]).to_numpy()
        rules = FailureRules(use_tool_wear_rule=False).fit(fit)
        reasons.iloc[score_idx] = rules.explain(score).to_numpy()
    return risk, type_proba, reasons


def main():
    ensure_dirs()
    bundle = joblib.load(MODELS_DIR / "failure_model.joblib")
    threshold, features = bundle["threshold"], bundle["features"]

    df = load_prepared()
    risk, type_proba, reasons = cross_score(df, bundle["model_name"], features)

    out = pd.DataFrame({
        "run_id": df["UDI"],
        "product_id": df["Product ID"],
        "product_variant": df["product_type"],
        "air_temp_c": (df["air_temp_k"] - 273.15).round(2),
        "process_temp_c": (df["process_temp_k"] - 273.15).round(2),
        "temp_difference_k": df["temp_diff_k"].round(2),
        "speed_rpm": df["speed_rpm"],
        "torque_nm": df["torque_nm"],
        "power_w": df["power_w"].round(0),
        "tool_wear_min": df["tool_wear_min"],
        "strain_nm_min": df["strain_nm_min"].round(0),
        "actual_failure": df["failure"],
        "actual_failure_type": df.apply(actual_type_text, axis=1),
        "risk_score": risk.round(4),
    })
    out["alarm"] = (out["risk_score"] >= threshold).astype(int)
    out["risk_level"] = np.select(
        [out["risk_score"] >= threshold, out["risk_score"] >= threshold / 2],
        ["Red", "Yellow"], default="Green")
    out["predicted_failure_type"] = np.where(
        out["alarm"] == 1, type_proba.idxmax(axis=1).map(TYPE_NAMES), "")
    out["alarm_reason"] = [
        "; ".join(REASON_TEXT[t] for t in reasons.columns if r[t]) for _, r in reasons.iterrows()]
    out["outcome"] = np.select(
        [(out["alarm"] == 1) & (out["actual_failure"] == 1),
         (out["alarm"] == 0) & (out["actual_failure"] == 1),
         (out["alarm"] == 1) & (out["actual_failure"] == 0)],
        ["Caught failure", "Missed failure", "False alarm"], default="Correct, no alarm")
    out["tool_wear_band"] = pd.cut(out["tool_wear_min"], bins=[-1, 50, 100, 150, 200, 260],
                                   labels=["0-50", "51-100", "101-150", "151-200", "201+"]).astype(str)
    out.to_csv(POWERBI_DIR / "machine_runs_scored.csv", index=False)

    # Test-set results and feature importance for the "model" page
    pd.read_csv(REPORTS_DIR / "test_results.csv").to_csv(POWERBI_DIR / "model_results.csv", index=False)
    importance = json.loads((REPORTS_DIR / "explainability.json").read_text())
    pd.DataFrame(importance["global_importance_mean_abs_shap"].items(),
                 columns=["feature", "importance"]).to_csv(POWERBI_DIR / "feature_importance.csv", index=False)

    summary = out["outcome"].value_counts().to_dict()
    summary["alarm_threshold"] = threshold
    summary["risk_levels"] = out["risk_level"].value_counts().to_dict()
    (REPORTS_DIR / "powerbi_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
