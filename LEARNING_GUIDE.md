# Learning Guide

This project was built for you to study afterwards. Work through it in the order below.
For every stage: **read the document, run the notebook, then answer the questions without
looking.** If you can explain every answer in your own words, you can defend this project
in any interview.

Rough time: 2–3 days at your pace.

---

## Setup (once)

1. Copy everything from this folder into your existing project folder
   `Desktop\kaizen-machine-failure-prediction` and replace files when asked.
   Your `.venv` and `data\raw\ai4i2020.csv` stay where they are.
2. In the VS Code terminal (with `(.venv)` active):
   ```
   python -m pip install -r requirements.txt
   python -m src.pipeline
   python -m pytest
   ```
   The pipeline takes about 2 minutes. You should see `8 passed` from the tests.
3. Open the notebooks in `notebooks/`, choose the `.venv` kernel, and use **Run All**.
4. Save to GitHub:
   ```
   git add .
   git commit -m "feat: complete predictive maintenance project"
   git push
   ```

---

## Stage 1–2: Problem and AI framing
**Read:** `docs/01_problem_definition.md`, `docs/02_ai_problem_framing.md`

**Questions**
1. Why is a missed failure more expensive than a false alarm?
2. Why did we choose classification instead of anomaly detection?
3. Why is RUL prediction impossible with this data?
4. Why must every ML model be compared with simple baselines?

<details><summary>Answers</summary>

1. A missed failure stops production, delays orders and needs emergency repair; a false alarm only costs a technician's check. We assumed 10:1.
2. We have labeled failures, and anomaly detection cannot name the failure type and usually raises more false alarms.
3. Each row is an independent snapshot without timestamps, so there is no sequence from healthy to failure.
4. If a model can't beat "always no failure" or simple rules, the simpler solution is better: cheaper, easier to trust and maintain.
</details>

---

## Stage 3: Data understanding
**Read:** `docs/03_data_understanding.md` · **Run:** `notebooks/01_data_understanding.ipynb`

**Questions**
1. Name the five trust checks.
2. Why is 96.6% accuracy meaningless here?
3. Why are there 373 failure-type marks but only 339 failures?
4. How did you spot that the data is synthetic?

<details><summary>Answers</summary>

1. Source, completeness, uniqueness, validity, consistency.
2. Only 3.39% of runs fail; a model always saying "no failure" reaches 96.6% while catching nothing.
3. 9 failures have no type, 18 random-failure marks belong to non-failures, and 24 runs have several types (25 extra marks).
4. Average process temperature is almost exactly 10 K above air temperature; real data is rarely that neat.
</details>

---

## Stage 4: EDA
**Read:** `docs/04_eda_findings.md` · **Run:** `notebooks/02_eda.ipynb` · **Code:** `src/eda.py`

**Questions**
1. Why is EDA done on the training data only?
2. Power correlates only 0.18 with failure. Why is it still important?
3. Describe the four failure mechanisms.
4. Why do L products fail more? Why is "L products are bad" a weak conclusion?

<details><summary>Answers</summary>

1. Ideas and rules derived from the test set would leak information into the model choice; the test set must stay unseen.
2. Failures occur at both very low and very high power; the two ends cancel out in a correlation, which only measures straight-line relationships.
3. Heat: small temperature difference and low speed. Power: too low or too high. Overstrain: wear × torque above a variant-specific limit. Tool wear: random moment after ~200 min.
4. Mostly overstrain, because their strain limit is lower. The useful conclusion is variant-specific torque limits or tool-change intervals.
</details>

---

## Stage 5: Preprocessing
**Read:** `docs/05_preprocessing.md` · **Code:** `src/data.py`

**Questions**
1. Which columns must never be inputs, and why?
2. Why did we keep the outliers?
3. Why class weights instead of SMOTE?
4. Why is the scaler inside a pipeline?

<details><summary>Answers</summary>

1. UDI and Product ID (identifiers) and TWF/HDF/PWF/OSF/RNF (they describe the failure itself → data leakage).
2. They are physically possible and exactly where failures happen.
3. Class weights invent no artificial data; synthetic samples near sharp physical limits could be misleading.
4. So it is fitted only on the training folds during cross-validation; fitting on all data would leak information.
</details>

---

## Stages 6–7: Modeling and evaluation
**Read:** `docs/06_modeling_and_evaluation.md` · **Run:** `notebooks/03_modeling.ipynb` · **Code:** `src/train.py`, `src/modeling.py`, `src/rules.py`

**Questions**
1. What is cross-validation, and why not compare models on the test set?
2. Why is the alarm threshold 0.40 and not 0.5?
3. Why is PR-AUC better than ROC-AUC here?
4. The expert rules are as good as the random forest. Is the ML model useless?
5. Most misses are tool-wear failures. What do you recommend?

<details><summary>Answers</summary>

1. Train on 4 folds, predict the 5th, repeat; every run gets an honest prediction. Comparing on the test set would slowly fit our choices to it.
2. It gave the lowest business cost on the training data; 0.5 is just a habit.
3. With rare failures, ROC-AUC looks high even for weaker models (0.970 vs 0.976), while PR-AUC shows the real gap (0.756 vs 0.894).
4. No. The synthetic data follows exact limits our EDA found. On real machines limits are unknown, noisy and drift; ML learns them from data. Choose on real data.
5. Planned tool changes at ~200 minutes; prediction for the physical failure modes.
</details>

---

## Stage 8: Explainability
**Read:** `docs/07_explainability.md` · **Run:** `notebooks/04_explainability.ipynb`

**Questions**
1. What does a SHAP value mean?
2. Why is the SHAP baseline 0.50 and not 0.034?
3. Explain the example alarm to a technician in one sentence.

<details><summary>Answers</summary>

1. How much one feature pushed one prediction up or down compared to the baseline.
2. Class weights make the model treat failures and normal runs as equally important, so its output is a risk score, not a calibrated probability.
3. "This tool is worn (192 min) and under high torque (65 Nm), so it's close to its overstrain limit: check the tool and reduce the load."
</details>

---

## Stage 9: Power BI dashboard
**Follow:** `docs/08_powerbi_dashboard.md`. Build the four pages yourself, check that your
measures match the expected totals, and add the screenshots to the README.

---

## Your 60-second interview pitch (practice out loud)
> "I built a predictive maintenance system for a CNC milling machine using the AI4I 2020
> dataset. I started with the business problem: a missed failure costs far more than a false
> alarm, so I optimized for business cost, not accuracy. In the data checks I found 27
> contradicting labels and runs with several failure types, which made my failure-type model
> multi-label. In the EDA I engineered physics-based features (power, temperature difference
> and tool strain) and rediscovered the four failure mechanisms. A random forest with those
> features catches 82% of failures with 93% precision on the test set, and every alarm is
> explained with SHAP. Most misses were tool-wear failures, which occur randomly after ~200
> minutes, so I recommended planned tool changes instead of prediction. I also found that
> simple expert rules perform as well on this synthetic data, and I documented when to choose
> rules vs. ML on a real machine."
