# 05 – Preprocessing and Feature Engineering

Code: `src/data.py`

## 1. What goes into the model
| Group | Columns |
|---|---|
| Product | `product_quality` (L = 0, M = 1, H = 2) |
| Sensors | air temperature, process temperature, rotational speed, torque, tool wear |
| Physics features | temperature difference, mechanical power, strain (tool wear × torque) |

**Never used as inputs:** `UDI`, `Product ID` (identifiers) and `TWF`, `HDF`, `PWF`,
`OSF`, `RNF`, which describe the failure itself (data leakage). A model using them
would look perfect in testing and be useless in the factory, where the failure type
is only known after the failure.

## 2. Encoding the product variant
L, M and H are **ordered** quality levels, so they are encoded as 0, 1, 2 instead of
three separate yes/no columns. This keeps the order information and one column fewer.

## 3. Cleaning
- **Missing values:** none, so no imputation.
- **Duplicates:** none.
- **Outliers:** the extreme speed and torque values are physically possible and are
  exactly where failures happen, so they are **kept**. Removing them would remove the
  signal we want to learn.
- **Label noise:** handled as decided in `03_data_understanding.md`.

## 4. Scaling
Only logistic regression needs scaled inputs (it compares feature sizes directly).
The scaler sits **inside a pipeline**, so during cross-validation it is fitted on the
training folds only. Tree-based models split on thresholds and need no scaling.

## 5. Class imbalance
Only 3.39% of runs fail. Options considered:

| Option | Decision |
|---|---|
| Oversampling (e.g. SMOTE) | Not used: creates artificial failures, risky near sharp physical limits |
| Undersampling | Not used: throws away 96% of the information about normal runs |
| **Class weights** | **Used:** each failure counts as much as ~29 normal runs, no data invented or lost |
| **Cost-based threshold** | **Used:** the alarm threshold is chosen by business cost (see `06_modeling_and_evaluation.md`) |

Side effect of class weights: the model output is a **risk score** that ranks runs
correctly, but it is not a calibrated real-world probability.

## 6. Train/test split
- 80% training (8,000 runs, 271 failures), 20% test (2,000 runs, 68 failures)
- **Stratified**, so both parts have the same 3.4% failure share
- Fixed random seed (42) for reproducibility
- The EDA, model selection and threshold choice use the **training data only**
