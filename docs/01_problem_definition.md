# 01 – Problem Definition

## 1. Situation
A CNC milling machine fails unexpectedly several times per month. Each
unplanned stop interrupts production, delays customer orders and causes
expensive emergency repairs.

## 2. Business problem
Unplanned machine failures cause downtime and late deliveries, because
the maintenance team has no early warning before a failure happens.

## 3. Stakeholders and what each one needs
- **Maintenance manager:** wants to know which machine is likely to fail
  and why, early enough to plan the repair and order spare parts in time.
- **Production manager:** wants fewer unplanned stops and reliable delivery
  dates, and wants maintenance scheduled in low-load periods instead of
  in the middle of urgent orders.
- **Machine operator:** needs a simple, clear signal (green / yellow / red)
  with a recommended action, and no flood of false alarms that would make
  them stop trusting the system.

## 4. Goal and success criteria
**Goal:** reduce unplanned downtime by warning the maintenance team before
a failure happens.

**Which mistake is more expensive?**
- A **missed failure** stops the line, delays orders and causes emergency repairs.
- A **false alarm** costs a technician some time to check a healthy machine.

We assume a missed failure costs roughly **10 times more** than a false alarm
(to be validated with the maintenance and finance teams). Therefore the
system must **catch as many real failures as possible**, while keeping
false alarms at a level technicians can handle.

**Success criteria (initial targets, to be agreed with stakeholders):**
- Detect at least **80% of real failures** before they happen (recall).
- Keep false alarms low enough that technicians still trust the system.
- Every alarm shows the **likely failure type** and the **main reasons**.
- The trade-off between missed failures and false alarms can be adjusted.

## 5. Available data
Source: **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning
Repository), a synthetic dataset designed to reflect real industrial
predictive maintenance data. 10,000 production runs.

For each production run:
- Product quality variant: L (low), M (medium), H (high)
- Air temperature [K] and process temperature [K]
- Rotational speed [rpm] and torque [Nm]
- Tool wear [min]

For each failure:
- Machine failure (yes / no)
- Failure type: tool wear (TWF), heat dissipation (HDF), power (PWF),
  overstrain (OSF), random (RNF)

Not available: timestamps, vibration data, maintenance history.

## 6. Technical requirements
### Functional (what the system must do)
- **FR1:** Predict the probability that a production run ends in a failure,
  based on the sensor values.
- **FR2:** Identify the most likely failure type.
- **FR3:** Explain the main factors behind each prediction.
- **FR4:** Provide a dashboard with machine health, alarms and maintenance
  KPIs (failure rate by type and by product variant).

### Non-functional (how well / under what conditions)
- **NFR1:** Prioritize catching failures (recall) over avoiding false alarms,
  with an adjustable alarm threshold.
- **NFR2:** Predictions must be understandable for maintenance technicians.
- **NFR3:** A prediction must take less than one second.
- **NFR4:** Results must be reproducible: versioned code, fixed random seeds,
  documented data preparation.

## 7. Constraints and risks
- **Rare failures:** only about 3–4% of runs end in a failure (to be verified
  in Stage 3). A model that always predicts "no failure" would look 96%
  accurate while being useless, so accuracy is the wrong metric.
- **Synthetic data:** results may not transfer directly to a real machine
  and must be validated on real plant data before production use.
- **No time series:** each row is a snapshot, so the system can predict
  failure risk, but not how much time is left (Remaining Useful Life).
- **Data leakage risk:** the failure-type columns directly reveal the answer
  and must never be used as model inputs.
- **Random failures (RNF)** are random by definition and cannot be predicted.

## 8. Out of scope
- Automatically stopping or controlling the machine.
- Remaining Useful Life (RUL) prediction.
- Live connection to the machine's PLC (possible future integration with
  the Kaizen Factory Stack data platform).
- Spare parts ordering and maintenance scheduling optimization.