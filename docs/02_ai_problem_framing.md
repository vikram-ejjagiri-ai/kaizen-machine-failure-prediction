# 02 – AI Problem Framing

## 1. From business question to AI question
**Business question:** Will this machine fail, and why?

**AI question:** Given the sensor values of a production run, what is the
probability that the run ends in a machine failure, and which failure type
is most likely?

- **Unit of prediction:** one production run (one row of data)
- **Inputs (features):** product variant (L/M/H), air temperature, process
  temperature, rotational speed, torque, tool wear
- **Output (target):** machine failure (yes/no) and failure type
- **Excluded from inputs:**
  - `UDI`, `Product ID`: identifiers with no physical meaning
  - `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: they describe the failure itself,
    so using them as inputs would be cheating (data leakage)

## 2. Possible solutions

| Option | How it works | Advantages | Disadvantages |
|---|---|---|---|
| A. Expert rules | Fixed thresholds, e.g. tool wear > 200 min | Simple, fully explainable, no training data needed | Misses combinations of factors, needs expert knowledge, never improves |
| B. Binary classification | Supervised model learns failure vs. no failure from labeled runs | Uses our labels, outputs a probability, adjustable alarm threshold | Failures are rare, so learning is harder |
| C. Failure-type classification | Supervised model predicts which failure type occurs | Tells technicians what to check | Even fewer examples per type; types can occur together |
| D. Anomaly detection | Unsupervised model learns "normal" and flags unusual runs | Needs no labels, can catch unknown problems | Cannot name the failure type, usually more false alarms, ignores the labels we have |
| E. RUL regression | Predicts the time left until failure | Best for planning | **Not possible:** the data has no time series |

## 3. Decision
1. **Main model: binary classification (B)** for failure risk (FR1).
   We have labeled failures, and a probability with an adjustable threshold
   fits the recall-first requirement (NFR1).
2. **Second model: failure-type classification (C)** for FR2, trained on
   failed runs only.
3. **Benchmarks the models must beat:**
   - **Dummy baseline:** always predict "no failure"
   - **Rule-based baseline (A):** simple thresholds derived from EDA

   If the ML model cannot clearly beat the rules, the rules are the better
   solution: simpler, cheaper and easier to trust.

Anomaly detection (D) is kept as a possible future extension for failure
modes that are not in the training data.

## 4. How we will measure success
Accuracy is misleading because failures are rare. We will use:
- **Recall:** share of real failures the model catches (most important)
- **Precision:** share of alarms that are real failures
- **Precision-recall curve / PR-AUC:** quality across all thresholds
- **Confusion matrix:** the four outcomes in numbers
- **Business cost:** missed failure = 10 cost units, false alarm = 1 cost
  unit (assumption from the problem definition); we choose the threshold
  with the lowest total cost

## 5. Validation strategy
- Hold out a **test set** (20%) that is used only once, at the very end
- Use a **stratified split**, so both sets have the same share of failures
- Compare models with **cross-validation** on the training set
- Fix random seeds so every result can be reproduced

## 6. Candidate models (simple first)
1. Logistic regression: simple, explainable reference model
2. Decision tree: easy-to-read rules
3. Random forest and gradient boosting: usually strongest on tabular data

## 7. Explainability
- Global: which features matter most overall (feature importance, SHAP)
- Local: why the model raised *this* specific alarm (SHAP per prediction)
