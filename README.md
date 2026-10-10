# Kaizen Machine Failure Prediction

Predicting machine failures on a CNC milling machine from temperature, speed, torque
and tool-wear data, and turning the predictions into maintenance decisions.

Built step by step like an industry project: problem definition → AI framing → data
trust checks → EDA → preprocessing → modeling → cost-based evaluation →
explainability → dashboard.

![Failure mechanisms](reports/figures/failure_mechanisms.png)

## Results (test set: 2,000 runs, 68 failures, used once)

| Model | Recall | Precision | Missed | False alarms | Business cost* |
|---|---|---|---|---|---|
| **Random forest + physics features** | **0.82** | **0.93** | 12 | 4 | 124 |
| Expert rules (heat, power, overstrain) | 0.82 | 1.00 | 12 | 0 | 120 |
| Always "no failure" | 0.00 | – | 68 | 0 | 680 |

\* missed failure = 10, false alarm = 1 (assumption documented in `docs/01_problem_definition.md`)

- **Physics-based features** (power, temperature difference, tool strain) lowered the
  cross-validated cost of the random forest from 663 to 445.
- The model catches **every power and overstrain failure** and 28 of 29 heat failures.
- **9 of the 12 misses are tool-wear failures**, which occur at a random moment after
  ~200 minutes. Recommendation: planned tool changes at ~200 min instead of prediction.
- Simple expert rules match the ML model on this **synthetic** dataset, because it was
  generated from physical limits; the EDA rediscovered them. See
  `docs/09_results_and_recommendations.md` for when to choose rules vs. ML on real data.
- Every alarm is explained with **SHAP** and the physical limit that was exceeded.

![Model comparison](reports/figures/model_comparison.png)

![Explained alarm](reports/figures/shap_example_alarm.png)

## Project stages

| Stage | Document | Notebook / code |
|---|---|---|
| 1. Problem definition | [docs/01](docs/01_problem_definition.md) | – |
| 2. AI problem framing | [docs/02](docs/02_ai_problem_framing.md) | – |
| 3. Data understanding | [docs/03](docs/03_data_understanding.md) | `notebooks/01_data_understanding.ipynb` |
| 4. EDA | [docs/04](docs/04_eda_findings.md) | `notebooks/02_eda.ipynb`, `src/eda.py` |
| 5. Preprocessing | [docs/05](docs/05_preprocessing.md) | `src/data.py` |
| 6–7. Modeling and evaluation | [docs/06](docs/06_modeling_and_evaluation.md) | `notebooks/03_modeling.ipynb`, `src/train.py` |
| 8. Explainability | [docs/07](docs/07_explainability.md) | `notebooks/04_explainability.ipynb`, `src/explain.py` |
| 9. Dashboard | [docs/08](docs/08_powerbi_dashboard.md) | `src/export_powerbi.py`, `powerbi/` |
| 10. Results | [docs/09](docs/09_results_and_recommendations.md) | – |

## Run it (Windows)

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Download the dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
and put `ai4i2020.csv` into `data/raw/`. Then:

```
python -m src.pipeline      # EDA, training, evaluation, explanations, dashboard data (~2 min)
python -m pytest            # automated checks
python -m src.predict --variant L --air 301.5 --process 310.6 --speed 1300 --torque 65 --wear 200
```

## Project structure
```
docs/        one document per project stage
notebooks/   step-by-step walkthroughs with outputs
src/         reusable code (data, rules, models, evaluation, explanations, export)
tests/       automated checks
reports/     figures and metrics
powerbi/     scored data for the Power BI dashboard
```

## Limitations
- The AI4I 2020 dataset is **synthetic**; results must be validated on real machine data.
- No timestamps, so no Remaining Useful Life prediction or trends over time.
- The test set contains only 68 failures, so metrics have wide uncertainty.
- The model output is a risk score (class-weighted), not a calibrated probability.

## Data source
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository.
