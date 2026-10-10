"""Central settings for the project.

Every number that a stakeholder could question (costs, split sizes, seeds)
lives here, in one place, instead of being hidden inside the code.
"""
from pathlib import Path

# --- Folders -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
POWERBI_DIR = ROOT / "powerbi"

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42          # same seed everywhere -> same results every run
TEST_SIZE = 0.20           # 20% of runs are locked away for the final test
CV_FOLDS = 5               # 5-fold cross-validation on the training set

# --- Business costs (from docs/01_problem_definition.md) ----------------
# Assumption to be validated with maintenance and finance:
# a missed failure is about 10x as expensive as a false alarm.
COST_MISSED_FAILURE = 10   # false negative
COST_FALSE_ALARM = 1       # false positive


def ensure_dirs():
    """Create output folders if they don't exist yet."""
    for folder in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, POWERBI_DIR):
        folder.mkdir(parents=True, exist_ok=True)
