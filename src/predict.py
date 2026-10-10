"""Score one production run from the command line - how an operator tool would use the model.

Example:
  python -m src.predict --variant L --air 301.5 --process 310.6 --speed 1300 --torque 65 --wear 200
"""
import argparse

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.data import add_model_columns, add_physics_features
from src.export_powerbi import REASON_TEXT, TYPE_NAMES
from src.train import type_probabilities


def score_run(variant, air_k, process_k, speed_rpm, torque_nm, tool_wear_min):
    bundle = joblib.load(MODELS_DIR / "failure_model.joblib")
    type_bundle = joblib.load(MODELS_DIR / "type_model.joblib")
    rules = joblib.load(MODELS_DIR / "expert_rules.joblib")

    run = pd.DataFrame([{
        "product_type": variant, "air_temp_k": air_k, "process_temp_k": process_k,
        "speed_rpm": speed_rpm, "torque_nm": torque_nm, "tool_wear_min": tool_wear_min}])
    run = add_model_columns(add_physics_features(run))

    risk = float(bundle["model"].predict_proba(run[bundle["features"]])[:, 1][0])
    threshold = bundle["threshold"]
    level = "RED" if risk >= threshold else ("YELLOW" if risk >= threshold / 2 else "GREEN")
    types = type_probabilities(type_bundle["model"], run[type_bundle["features"]]).iloc[0]
    reasons = [REASON_TEXT[t] for t, hit in rules.explain(run).iloc[0].items() if hit]
    return {"risk_score": risk, "risk_level": level, "threshold": threshold,
            "likely_type": TYPE_NAMES[types.idxmax()], "rule_reasons": reasons,
            "power_w": float(run["power_w"][0]), "strain": float(run["strain_nm_min"][0])}


def main():
    p = argparse.ArgumentParser(description="Failure risk for one production run")
    p.add_argument("--variant", choices=["L", "M", "H"], required=True)
    p.add_argument("--air", type=float, required=True, help="air temperature [K]")
    p.add_argument("--process", type=float, required=True, help="process temperature [K]")
    p.add_argument("--speed", type=float, required=True, help="rotational speed [rpm]")
    p.add_argument("--torque", type=float, required=True, help="torque [Nm]")
    p.add_argument("--wear", type=float, required=True, help="tool wear [min]")
    a = p.parse_args()

    r = score_run(a.variant, a.air, a.process, a.speed, a.torque, a.wear)
    print(f"Risk score : {r['risk_score']:.2f}  (alarm at >= {r['threshold']:.2f})")
    print(f"Risk level : {r['risk_level']}")
    if r["risk_level"] != "GREEN":
        print(f"Likely type: {r['likely_type']}")
    print(f"Power      : {r['power_w']:.0f} W | Strain: {r['strain']:.0f}")
    print("Reasons    : " + ("; ".join(r["rule_reasons"]) if r["rule_reasons"] else "no rule limit exceeded"))


if __name__ == "__main__":
    main()
