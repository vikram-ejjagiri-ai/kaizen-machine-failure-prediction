"""Run the whole project from raw data to dashboard data with one command.

Run:  python -m src.pipeline
"""
from src import eda, explain, export_powerbi, train


def main():
    steps = [("Stage 4 - EDA", eda.main),
             ("Stages 5-7 - Train and evaluate", train.main),
             ("Stage 8 - Explainability", explain.main),
             ("Stage 9 - Power BI data", export_powerbi.main)]
    for name, step in steps:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        step()
    print("\nDone. See reports/ for figures and metrics, powerbi/ for dashboard data.")


if __name__ == "__main__":
    main()
