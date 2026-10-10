"""Loading and preparing the AI4I 2020 predictive maintenance dataset."""
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA, RANDOM_STATE, TEST_SIZE

DATA_PATH = RAW_DATA

# Short, code-friendly column names
RENAME = {
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}

SENSORS = ["air_temp_k", "process_temp_k", "speed_rpm", "torque_nm", "tool_wear_min"]
PHYSICS_FEATURES = ["temp_diff_k", "power_w", "strain_nm_min"]
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Failure types the type-model predicts. RNF is random by definition
# (docs/03_data_understanding.md, decision 4), so it is not modelled.
PREDICTABLE_TYPES = ["TWF", "HDF", "PWF", "OSF"]

# Product quality is ordered: L < M < H
PRODUCT_QUALITY = {"L": 0, "M": 1, "H": 2}

# The two feature sets we compare (does physics-based engineering help?)
FEATURES_RAW = ["product_quality"] + SENSORS
FEATURES_PHYSICS = FEATURES_RAW + PHYSICS_FEATURES

# Columns that must NEVER be model inputs (identifiers and data leakage)
NOT_INPUTS = ["UDI", "Product ID", "failure"] + FAILURE_TYPES


def load_data(path=DATA_PATH):
    """Load the raw CSV with short column names."""
    return pd.read_csv(path).rename(columns=RENAME)


def add_physics_features(df):
    """Add features based on the physics of the milling process.

    temp_diff_k   : process minus air temperature. Small -> heat is not carried away.
    power_w       : mechanical power = torque x angular speed.
    strain_nm_min : tool wear x torque. A worn tool under high load is overstrained.
    """
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    df["power_w"] = df["torque_nm"] * df["speed_rpm"] * 2 * math.pi / 60
    df["strain_nm_min"] = df["tool_wear_min"] * df["torque_nm"]
    return df


def add_model_columns(df):
    """Encode product type as an ordered number the models can use."""
    df = df.copy()
    df["product_quality"] = df["product_type"].map(PRODUCT_QUALITY)
    return df


def load_prepared():
    """Raw data + physics features + encoded product quality."""
    return add_model_columns(add_physics_features(load_data()))


def split_train_test(df):
    """Stratified split: both parts keep the same 3.39% failure share.

    The test set is used exactly once, at the very end.
    """
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["failure"],
        random_state=RANDOM_STATE,
    )
