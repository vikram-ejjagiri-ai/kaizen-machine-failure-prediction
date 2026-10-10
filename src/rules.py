"""Rule-based baseline: the model every ML model must beat.

Maintenance experts often work with simple limits ("replace the tool after
200 minutes"). Here we turn the failure mechanisms found in the EDA into such
rules. To keep the comparison fair, every threshold is learned from the
TRAINING data only - the test set never influences the rules.
"""
import pandas as pd


class FailureRules:
    """Physics-based alarm rules for the four predictable failure types."""

    def __init__(self, use_tool_wear_rule=False):
        # The tool-wear rule raises many alarms (tool wear failures happen at a
        # random moment after ~200 min), so it is optional and compared separately.
        self.use_tool_wear_rule = use_tool_wear_rule
        self.thresholds_ = {}

    def fit(self, train):
        t = {}

        # Heat dissipation: small temperature difference AND low speed
        hdf = train[train["HDF"] == 1]
        t["hdf_max_temp_diff_k"] = hdf["temp_diff_k"].max()
        t["hdf_max_speed_rpm"] = hdf["speed_rpm"].max()

        # Power failure: power too low OR too high (two separate clusters)
        pwf = train[train["PWF"] == 1]
        middle = train["power_w"].median()
        t["pwf_low_power_w"] = pwf.loc[pwf["power_w"] < middle, "power_w"].max()
        t["pwf_high_power_w"] = pwf.loc[pwf["power_w"] > middle, "power_w"].min()

        # Overstrain: strain limit depends on the product variant
        osf = train[train["OSF"] == 1]
        t["osf_min_strain"] = osf.groupby("product_type")["strain_nm_min"].min().to_dict()

        # Tool wear: wear limit where tool wear failures start
        twf = train[train["TWF"] == 1]
        t["twf_min_tool_wear_min"] = twf["tool_wear_min"].min()

        self.thresholds_ = t
        return self

    def explain(self, df):
        """One True/False column per rule, so every alarm has a reason."""
        t = self.thresholds_
        reasons = pd.DataFrame(index=df.index)
        reasons["HDF"] = (df["temp_diff_k"] <= t["hdf_max_temp_diff_k"]) & (
            df["speed_rpm"] <= t["hdf_max_speed_rpm"])
        reasons["PWF"] = (df["power_w"] <= t["pwf_low_power_w"]) | (
            df["power_w"] >= t["pwf_high_power_w"])
        osf_limit = df["product_type"].map(t["osf_min_strain"])
        reasons["OSF"] = df["strain_nm_min"] >= osf_limit
        reasons["TWF"] = (df["tool_wear_min"] >= t["twf_min_tool_wear_min"]) & self.use_tool_wear_rule
        return reasons

    def predict(self, df):
        """1 = raise an alarm, 0 = no alarm."""
        return self.explain(df).any(axis=1).astype(int).to_numpy()
