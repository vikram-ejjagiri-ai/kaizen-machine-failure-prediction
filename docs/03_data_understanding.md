# 03 – Data Understanding

Notebook: `notebooks/01_data_understanding.ipynb`

## 1. Source
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
A synthetic dataset created at HTW Berlin to reflect real industrial
predictive maintenance data. 10,000 production runs, 14 columns.

Fingerprint of synthetic data: the average process temperature (310.01 K)
is almost exactly 10 K above the average air temperature (300.00 K).
Results must therefore be validated on real machine data before use in production.

## 2. Trust checks

| Check | Result | Verdict |
|---|---|---|
| Completeness | 0 missing values in all columns | OK |
| Uniqueness | 0 duplicate rows; `UDI` and `Product ID` unique | OK |
| Validity | All values physically plausible (see section 3) | OK |
| Consistency | 27 rows with contradicting labels (see section 5) | Needs a decision |
| Balance | Only 3.39% of runs are failures | Imbalanced |

## 3. Value ranges

| Variable | Min | Median | Max | Comment |
|---|---|---|---|---|
| Air temperature [K] | 295.3 | 300.1 | 304.5 | 22.2–31.4 °C, normal factory hall |
| Process temperature [K] | 305.7 | 310.1 | 313.8 | Always hotter than air |
| Rotational speed [rpm] | 1168 | 1503 | 2886 | Long tail of very high speeds |
| Torque [Nm] | 3.8 | 40.1 | 76.6 | Some very low and very high values |
| Tool wear [min] | 0 | 108 | 253 | |
| Mechanical power [W] (derived) | 1148 | – | 10470 | Torque × angular speed; always positive |

The extreme speed and torque values are physically possible and are kept.
They will be examined in the EDA, since failures often occur at extremes.

## 4. Target variable
- **339 of 10,000 runs failed (3.39%).**
- A model that always predicts "no failure" would reach **96.61% accuracy**
  while catching zero failures, so accuracy is not used as a metric.

| Failure type | Count |
|---|---|
| Tool wear failure (TWF) | 46 |
| Heat dissipation failure (HDF) | 115 |
| Power failure (PWF) | 95 |
| Overstrain failure (OSF) | 98 |
| Random failure (RNF) | 19 |
| **Total type marks** | **373** |

## 5. Label inconsistencies
The 373 type marks do not match the 339 failures:
- **9 runs** are marked as failures, but no failure type is given.
- **18 runs** have a failure type marked (all 18 are random failures, RNF:
  18 of the 19 RNF cases), but are not marked as failures.
- **24 runs** have more than one failure type at the same time.

## 6. Product variants

| Variant | Share of runs | Failure rate |
|---|---|---|
| L (low quality) | 60% | 3.92% |
| M (medium quality) | 30% | 2.77% |
| H (high quality) | 10% | 2.09% |

Low-quality products fail almost twice as often as high-quality ones.
This is an association, not yet a cause; it is investigated in the EDA.

## 7. Decisions
1. **No imputation or deduplication needed:** the data is complete and unique.
2. **Binary target = `Machine failure` as given.** The 9 failures without a type
   remain failures; the 18 runs with a type but no failure remain non-failures.
   Both are documented as label noise.
3. **Failure-type model is multi-label** (one yes/no output per type), because
   24 runs have several types at once. The 9 failures without a type are
   excluded from this model.
4. **Random failures (RNF)** are random by definition and are not modeled as a
   separate type; some missed failures are therefore expected.
5. **Excluded from model inputs:** `UDI`, `Product ID` (identifiers) and
   `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (data leakage).
6. **Stratified train/test split** is mandatory because of the 3.39% failure rate.
