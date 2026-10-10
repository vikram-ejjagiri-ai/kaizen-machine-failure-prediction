"""Candidate models, metrics and the cost-based alarm threshold."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (COST_FALSE_ALARM, COST_MISSED_FAILURE, CV_FOLDS,
                        RANDOM_STATE)


def candidate_models():
    """Simple first, more powerful later (docs/02_ai_problem_framing.md, section 6).

    class_weight="balanced" tells each model that a failure counts as much as
    ~29 normal runs, so it cannot simply ignore the rare failures.
    """
    return {
        "Dummy (always 'no failure')": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Decision tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, class_weight="balanced",
            random_state=RANDOM_STATE),
    }


def cv_splitter():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def out_of_fold_probabilities(model, X, y):
    """Probability for every training run, each predicted by a model that never saw it.

    These honest probabilities are used to compare models and to choose the
    alarm threshold - without ever touching the test set.
    """
    return cross_val_predict(model, X, y, cv=cv_splitter(), method="predict_proba")[:, 1]


def business_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(fn * COST_MISSED_FAILURE + fp * COST_FALSE_ALARM)


def scores_at(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "caught_failures": int(tp),
        "missed_failures": int(fn),
        "false_alarms": int(fp),
        "business_cost": business_cost(y_true, y_pred),
    }


def ranking_scores(y_true, proba):
    """Threshold-free quality: how well the model ranks failures above normal runs."""
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def cost_curve(y_true, proba, thresholds=None):
    """Total business cost for every possible alarm threshold."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [business_cost(y_true, (proba >= t).astype(int)) for t in thresholds]
    return np.asarray(thresholds), np.asarray(costs)


def best_threshold(y_true, proba):
    """The alarm threshold with the lowest total business cost.

    If several thresholds tie, take the highest one: same cost, fewer alarms.
    """
    thresholds, costs = cost_curve(y_true, proba)
    best = np.flatnonzero(costs == costs.min()).max()
    return float(thresholds[best]), int(costs[best])
