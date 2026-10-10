# 08 – Power BI Dashboard

Data: `powerbi/` (created by `python -m src.export_powerbi`) · Requirement FR4

The data is ready; you build the dashboard yourself in **Power BI Desktop** (free, Windows).
Building it by hand is part of the learning, and it lets you say honestly in interviews:
*"I built this dashboard."*

## 1. The data

| File | One row per | Used for |
|---|---|---|
| `machine_runs_scored.csv` | production run (10,000) | everything |
| `model_results.csv` | model on the test set | model page |
| `feature_importance.csv` | feature | model page |

Key columns in `machine_runs_scored.csv`:

| Column | Meaning |
|---|---|
| `risk_score` | 0–1, from a model that never saw that run (5-fold cross-scoring) |
| `alarm` | 1 if `risk_score` ≥ 0.40 |
| `risk_level` | Red (≥ 0.40), Yellow (≥ 0.20), Green |
| `predicted_failure_type` | most likely type, filled for alarms |
| `alarm_reason` | physical limit exceeded, in plain words |
| `actual_failure`, `actual_failure_type` | what really happened |
| `outcome` | Caught failure / Missed failure / False alarm / Correct, no alarm |
| `tool_wear_band` | 0-50, 51-100, 101-150, 151-200, 201+ |

Expected totals (check your measures against these): 10,000 runs, 339 failures,
285 caught, 54 missed, 31 false alarms, 316 red runs.

The data has **no timestamps**, so the dashboard shows the machine's condition across
production runs, not trends over time. Don't add a time axis.

## 2. Load the data
1. Install **Power BI Desktop** (Microsoft Store or microsoft.com/power-bi).
2. **Home → Get data → Text/CSV** → `machine_runs_scored.csv` → **Transform Data**.
3. In Power Query, check the column types: `risk_score` = Decimal number, `alarm` and
   `actual_failure` = Whole number, text columns = Text. → **Close & Apply**.
4. Repeat for `model_results.csv` and `feature_importance.csv` (**Load** is enough).

## 3. Measures (DAX)
Select the `machine_runs_scored` table → **Modeling → New measure**, one at a time:

```DAX
Total Runs = COUNTROWS(machine_runs_scored)

Failures = SUM(machine_runs_scored[actual_failure])

Failure Rate = DIVIDE([Failures], [Total Runs])

Alarms = SUM(machine_runs_scored[alarm])

Caught Failures =
    CALCULATE([Total Runs], machine_runs_scored[outcome] = "Caught failure")

Missed Failures =
    CALCULATE([Total Runs], machine_runs_scored[outcome] = "Missed failure")

False Alarms =
    CALCULATE([Total Runs], machine_runs_scored[outcome] = "False alarm")

Recall = DIVIDE([Caught Failures], [Failures])

Precision = DIVIDE([Caught Failures], [Alarms])

Business Cost = [Missed Failures] * 10 + [False Alarms] * 1

Red Runs = CALCULATE([Total Runs], machine_runs_scored[risk_level] = "Red")
```

Format `Failure Rate`, `Recall` and `Precision` as **Percentage** (Measure tools → %).

**Sort risk levels correctly** (otherwise Power BI sorts alphabetically: Green, Red, Yellow):
**Modeling → New column**:
```DAX
Risk Order = SWITCH(machine_runs_scored[risk_level], "Green", 1, "Yellow", 2, "Red", 3)
```
Then select `risk_level` → **Column tools → Sort by column → Risk Order**.

## 4. Page 1 – Machine health overview (for the production manager)
- **Cards:** Total Runs, Failure Rate, Alarms, Red Runs
- **Donut chart:** Legend `risk_level`, Values `Total Runs`. Set colors manually:
  Green `#0ca30c`, Yellow `#fab219`, Red `#d03b3b`, and turn on **data labels with category
  names**, so color is never the only signal.
- **Clustered bar chart:** Y-axis `product_variant`, X-axis `Failure Rate`
- **Bar chart:** Y-axis `actual_failure_type`, X-axis `Failures`
  (filter out "No failure" in the Filters pane)
- **Slicers:** `product_variant`, `tool_wear_band`

## 5. Page 2 – Maintenance alarms (for the maintenance team)
- **Table:** `run_id`, `product_variant`, `risk_score`, `predicted_failure_type`,
  `alarm_reason`, `tool_wear_min`, `torque_nm`.
  Filter: `risk_level` = Red. Sort by `risk_score` descending.
  Add **conditional formatting** on `risk_score` (background color scale).
- **Scatter chart:** X `tool_wear_min`, Y `torque_nm`, Legend `risk_level`
  (shows the overstrain zone: worn tool + high torque)
- **Bar chart:** Y `predicted_failure_type`, X `Alarms`

## 6. Page 3 – Model performance (for engineers and reviewers)
- **Cards:** Recall, Precision, Missed Failures, False Alarms, Business Cost
- **Matrix (confusion matrix):** Rows `actual_failure`, Columns `alarm`, Values `Total Runs`
- **Bar chart:** Y `actual_failure_type`, X `Missed Failures` → shows that most misses
  are tool-wear failures
- **Table:** all columns of `model_results`
- **Bar chart:** `feature_importance` (Y `feature`, X `importance`, sorted descending)

## 7. Page 4 (advanced) – What-if: move the alarm threshold
Lets a manager see the trade-off live.
1. **Modeling → New parameter → Numeric range**: name `Threshold`, min 0.05, max 0.95,
   increment 0.05, default 0.40, tick "Add slicer to this page".
   Power BI creates a measure called `Threshold Value`.
2. New measures:
```DAX
Caught @ Threshold =
    CALCULATE([Total Runs],
        FILTER(machine_runs_scored,
            machine_runs_scored[actual_failure] = 1
            && machine_runs_scored[risk_score] >= [Threshold Value]))

False Alarms @ Threshold =
    CALCULATE([Total Runs],
        FILTER(machine_runs_scored,
            machine_runs_scored[actual_failure] = 0
            && machine_runs_scored[risk_score] >= [Threshold Value]))

Missed @ Threshold = [Failures] - [Caught @ Threshold]

Cost @ Threshold = [Missed @ Threshold] * 10 + [False Alarms @ Threshold]
```
3. Show the four measures as **cards** next to the slider. Move the slider and watch
   missed failures and false alarms trade places.

## 8. Finish
- Give every page a title and a one-line explanation of who it's for.
- **File → Save** as `powerbi/predictive_maintenance.pbix`.
- Take screenshots of each page for the README and your portfolio
  (save them as `reports/figures/powerbi_page1.png` etc.).
