"""
Steps 2 (feature selection), 3 (model) and 4 (evaluation).

Design decisions
----------------
* Target     : Risk level (Low / Medium / High) defined in risk_labeling.py.
* Leakage    : Grade and Total_Score are NOT features (Grade is part of the label and
               would not exist yet for a student we want to warn *early*).
* Fairness   : Gender is deliberately excluded - a risk model must not flag students
               because of who they are. Department is excluded because EDA showed
               no meaningful difference between departments.
* Selection  : candidate numeric features are ranked by Random-Forest importance on the
               TRAINING split only; those above the mean importance are kept.
* Models     : Logistic Regression, Decision Tree, Random Forest, Gradient Boosting are
               compared with 5-fold stratified CV (macro-F1) and the winner is scored once
               on a held-out 20 % test set.
* Imbalance  : High risk is the rarest class but the most important to catch, so
               models use class weights and I report per-class recall.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg

CANDIDATE_FEATURES = [
    "Attendance", "Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
    "Participation_Score", "Projects_Score", "Marks", "Age", "test_preparation_course",
    "math_score", "reading_score", "writing_score", "science_score",
]


def split_data(df: pd.DataFrame):
    X, y = df[CANDIDATE_FEATURES], df["Risk"].astype(str)
    return train_test_split(X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)


def select_features(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    """Rank features by Random-Forest importance (training data only). Returns the full ranking."""
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                random_state=cfg.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train.fillna(X_train.median()), y_train)
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)


def build_models() -> dict:
    """Each model gets the same preprocessing: median-impute (for future incomplete records) + scale."""
    def pipe(clf):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])
    rs = cfg.RANDOM_STATE
    return {
        "Baseline (majority class)": pipe(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                                     class_weight="balanced", random_state=rs)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                     class_weight="balanced_subsample", random_state=rs, n_jobs=-1)),
        "Gradient Boosting": pipe(GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                                                             random_state=rs)),
    }


def compare_models(models: dict, X_train, y_train) -> pd.DataFrame:
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = {"accuracy": "accuracy",
               # zero_division=0 -> the majority-class baseline never predicts Medium/High; avoids warnings
               "precision": make_scorer(precision_score, average="macro", zero_division=0),
               "recall": "recall_macro", "f1": "f1_macro"}
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"Model": name, **{k: s[f"test_{k}"].mean() for k in scoring},
                     "f1_std": s["test_f1"].std()})
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def evaluate(model, X_test, y_test, name: str) -> tuple:
    """Held-out evaluation: accuracy, precision, recall, F1, confusion matrix."""
    pred = model.predict(X_test)
    labels = cfg.RISK_ORDER
    cm = confusion_matrix(y_test, pred, labels=labels)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "per_class": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
    }
    text = classification_report(y_test, pred, labels=labels, digits=3, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set(xlabel="Predicted risk", ylabel="Actual risk", title=f"Confusion matrix - {name}")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".0%", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set(xlabel="Predicted risk", ylabel="Actual risk", title="Row-normalised (= recall per class)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "06_confusion_matrix.png", dpi=130)
    plt.close(fig)
    return result, text


def plot_importance(model, X_test, y_test, name: str) -> pd.Series:
    """Permutation importance on the test set: how much does macro-F1 drop when a feature is shuffled?"""
    pi = permutation_importance(model, X_test, y_test, scoring="f1_macro", n_repeats=8,
                                random_state=cfg.RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    ax.set_title(f"Feature importance - {name}")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "07_feature_importance.png", dpi=130)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def outcome_only_check(df: pd.DataFrame) -> dict:
    """
    Honest sanity check. The Risk label is partly built from Attendance and Marks, so the
    model above can partly "re-discover the rule". To measure how well we can predict a
    REAL outcome - failing (grade D/F) - from information available during the semester,
    train on the observed grade alone. ROC-AUC: 0.5 = coin flip, 1.0 = perfect.
    """
    y = df["Grade"].isin(["D", "F"]).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    res = {"share_DF": float(y.mean())}
    X_all = df[CANDIDATE_FEATURES]
    pipe = lambda clf: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("c", clf)])
    res["attendance_only_auc"] = float(cross_val_score(
        pipe(LogisticRegression(max_iter=1000)), df[["Attendance"]], y, cv=cv, scoring="roc_auc").mean())
    res["all_features_logreg_auc"] = float(cross_val_score(
        pipe(LogisticRegression(max_iter=1000)), X_all, y, cv=cv, scoring="roc_auc").mean())
    res["all_features_rf_auc"] = float(cross_val_score(
        pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=cfg.RANDOM_STATE, n_jobs=-1)),
        X_all, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
    return res


def save_json(obj, name):
    (cfg.REPORT_DIR / name).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
