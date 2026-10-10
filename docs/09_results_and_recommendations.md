# 09 – Results and Recommendations

## 1. Answer to the business problem
| Requirement | Target | Result (test set, 2,000 runs, 68 failures) | Status |
|---|---|---|---|
| FR1 – predict failure risk | – | Risk score per run, alarm at 0.40 | Done |
| Catch failures (recall) | ≥ 80% | **82%** (56 of 68) | Met |
| Few false alarms | trusted by technicians | **4 false alarms** (precision 93%) | Met |
| FR2 – failure type | – | Top type correct for **97%** of failures | Done |
| FR3 – explain each alarm | – | SHAP per prediction + physical limit exceeded | Done |
| FR4 – dashboard | – | Power BI data and build guide (`08_powerbi_dashboard.md`) | Ready to build |
| NFR3 – fast | < 1 s | Milliseconds per run | Met |
| NFR4 – reproducible | – | Fixed seed, one command (`python -m src.pipeline`) | Met |

## 2. Main findings
1. **Three of four failure mechanisms follow sharp physical limits** (heat dissipation,
   power, overstrain). The model catches all power and overstrain failures and 28 of 29
   heat failures on the test set.
2. **Physics-based features matter more than the algorithm.** They lowered the random
   forest's business cost from 663 to 445 in cross-validation.
3. **Tool-wear failures cannot be predicted from a snapshot.** They occur at a random
   moment after ~200 minutes of wear; 9 of the 12 missed test failures are tool wear.
4. **Simple expert rules perform as well as the best ML model on this dataset**, because
   the synthetic data was generated from such limits.
5. **L products fail more mainly because of overstrain:** their strain limit is lower.

## 3. Recommendations for the plant
1. **Use the model (or the rules) for heat, power and overstrain alarms**, with the
   reason shown to the technician.
2. **Handle tool wear with maintenance planning, not prediction:** schedule a tool change
   at about **200 minutes** of wear. The alarm system and the maintenance plan complement
   each other.
3. **Make torque limits or tool-change intervals depend on the product variant**, since
   L products tolerate less strain.
4. **Choose between rules and ML on real data.** On a real machine, compare both again: if
   the physical limits are known and stable, rules are simpler to run and audit; if they
   are unknown, noisy or drifting, the ML model learns them from data.
5. **Validate the cost assumption** (missed failure = 10× false alarm) with maintenance and
   finance, and re-tune the threshold. A slightly lower threshold (≈ 0.35) gives more margin.

## 4. Model card
| | |
|---|---|
| **Model** | Random forest, 400 trees, class-weighted, scikit-learn |
| **Inputs** | Product quality, 5 sensor values, 3 physics features |
| **Output** | Risk score 0–1 (ranking score, not a calibrated probability); alarm at ≥ 0.40 |
| **Training data** | 8,000 runs of the synthetic AI4I 2020 dataset |
| **Test performance** | Recall 0.82, precision 0.93, PR-AUC 0.876, ROC-AUC 0.978 |
| **Intended use** | Decision support for maintenance planning on one machine type |
| **Not intended for** | Automatic machine shutdown; predicting time to failure; other machines without retraining |
| **Known limitations** | Synthetic data; misses most tool-wear failures; ignores random failures; no time series; small test set (68 failures), so metrics have wide uncertainty |

## 5. Next steps
- Validate on real machine data; connect to the **Kaizen Factory Stack** (OPC UA → MQTT →
  PostgreSQL) to score production runs live.
- Add timestamps and maintenance history to enable Remaining Useful Life prediction
  (see the NASA C-MAPSS project).
- Monitor for **data drift**: if sensor distributions change, retrain and re-check the threshold.
- Serve the model as a small API (FastAPI) so dashboards and MES systems can call it.
