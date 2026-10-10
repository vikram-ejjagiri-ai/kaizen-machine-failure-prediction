"""Automated checks. Run:  python -m pytest

Tests protect against silent mistakes: a wrong formula or a leaked column
would not crash the code, it would just give wrong results.
"""
import math

import pandas as pd
import pytest

from src.config import RAW_DATA
from src.data import (FAILURE_TYPES, FEATURES_PHYSICS, FEATURES_RAW,
                      add_model_columns, add_physics_features, load_prepared,
                      split_train_test)
from src.modeling import best_threshold, business_cost
from src.rules import FailureRules

needs_data = pytest.mark.skipif(not RAW_DATA.exists(), reason="dataset not downloaded")


def one_run(**overrides):
    run = {"product_type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
           "speed_rpm": 1500.0, "torque_nm": 40.0, "tool_wear_min": 100.0}
    run.update(overrides)
    return add_model_columns(add_physics_features(pd.DataFrame([run])))


# --- Feature engineering ---------------------------------------------------
def test_physics_features_are_correct():
    run = one_run()
    assert run["temp_diff_k"][0] == pytest.approx(10.0)
    assert run["power_w"][0] == pytest.approx(40 * 1500 * 2 * math.pi / 60)
    assert run["strain_nm_min"][0] == pytest.approx(4000.0)


def test_product_quality_is_ordered():
    assert [one_run(product_type=t)["product_quality"][0] for t in "LMH"] == [0, 1, 2]


def test_no_leakage_columns_in_features():
    forbidden = set(FAILURE_TYPES) | {"failure", "UDI", "Product ID"}
    assert not forbidden & set(FEATURES_RAW)
    assert not forbidden & set(FEATURES_PHYSICS)


# --- Business cost -----------------------------------------------------------
def test_business_cost_weights():
    # 1 missed failure (10) + 2 false alarms (2) = 12
    assert business_cost([1, 0, 0, 1], [0, 1, 1, 1]) == 12


def test_best_threshold_prefers_catching_failures():
    y = [0, 0, 0, 1]
    proba = pd.Series([0.1, 0.2, 0.3, 0.25]).to_numpy()
    threshold, cost = best_threshold(y, proba)
    assert cost == 1          # catch the failure, accept one false alarm
    assert threshold <= 0.25


# --- Data and split ----------------------------------------------------------
@needs_data
def test_dataset_matches_documented_numbers():
    df = load_prepared()
    assert len(df) == 10_000
    assert df["failure"].sum() == 339
    assert df.isna().sum().sum() == 0


@needs_data
def test_split_is_stratified():
    train, test = split_train_test(load_prepared())
    assert len(train) == 8000 and len(test) == 2000
    assert abs(train["failure"].mean() - test["failure"].mean()) < 0.002


# --- Expert rules --------------------------------------------------------------
@needs_data
def test_rules_flag_clear_physical_failures():
    train, _ = split_train_test(load_prepared())
    rules = FailureRules().fit(train)
    overheated = one_run(air_temp_k=302.0, process_temp_k=310.0, speed_rpm=1300)  # diff 8 K
    overstrained = one_run(product_type="L", torque_nm=60, tool_wear_min=200)     # strain 12,000
    healthy = one_run()
    assert rules.predict(overheated)[0] == 1
    assert rules.predict(overstrained)[0] == 1
    assert rules.predict(healthy)[0] == 0
