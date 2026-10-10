# Portfolio and CV update

Every sentence below matches what this project actually contains.

## 1. Portfolio card (`src/data/projects.js` in your portfolio)

Replace the existing `predictive-maintenance` entry with this one (keep its `number`):

```js
  {
    id: "predictive-maintenance",
    number: "02",
    title: "Predictive Maintenance: Machine Failure Prediction",
    shortTitle: "Predictive Maintenance",
    category: "Industrial AI",
    tags: ["Machine Learning", "Explainable AI", "Power BI"],
    description:
      "A machine failure prediction system for a CNC milling process: physics-based features, cost-optimized alarms and explanations a maintenance technician can act on.",
    objective:
      "Predict machine failures from temperature, speed, torque and tool-wear data, identify the likely failure type, and support maintenance planning with explainable alarms.",
    achievements: [
      "Defined the business problem and technical requirements, prioritizing missed failures (10x cost) over false alarms.",
      "Audited data quality and found 27 contradicting labels and 24 multi-type failures, which shaped a multi-label failure-type model.",
      "Engineered physics-based features (power, temperature difference, tool strain) that cut the model's business cost by about a third.",
      "Achieved 82% recall at 93% precision on a held-out test set with a random forest, benchmarked against expert rules and a dummy baseline.",
      "Explained every alarm with SHAP and the physical limit exceeded; recommended planned tool changes for the random tool-wear failures the model cannot predict.",
      "Prepared scored data and a Power BI dashboard design for machine health, alarms and model performance."
    ],
    tools: ["Python", "Scikit-learn", "SHAP", "Pandas", "Power BI", "DAX", "Pytest"],
    visual: "sensor",
    image: "projects/predictive-maintenance.png",
    github: "https://github.com/vikram-ejjagiri-ai/kaizen-machine-failure-prediction"
  },
```

**Two things to do so it stays true:**
- After you **build** the Power BI dashboard, change the last achievement to:
  `"Built an interactive Power BI dashboard for machine health, maintenance alarms and model performance."`
- Copy a figure into the portfolio as the card image, e.g.
  `Copy-Item reports\figures\failure_mechanisms.png ..\portfolio\public\projects\predictive-maintenance.png`
  (or a Power BI screenshot once the dashboard exists).

What changed compared to the old card: the old text mentioned **vibration data**, which this
dataset doesn't contain, and a finished Power BI dashboard, which you still need to build.

## 2. CV bullet points
- Developed an explainable machine failure prediction system (Python, scikit-learn, SHAP) achieving 82% recall at 93% precision on held-out data, with alarm thresholds optimized for business cost.
- Engineered physics-based features (mechanical power, tool strain, heat dissipation) that reduced the cost of model errors by about a third versus raw sensor data.
- Identified label inconsistencies and an unpredictable failure mode during data analysis, and translated the findings into maintenance recommendations (variant-specific strain limits, planned tool changes).

"About a third": cross-validated business cost dropped from 663 to 445 (−33%).
