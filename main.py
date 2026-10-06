"""
Student academic-risk prediction - end-to-end pipeline.

Run:  python main.py
Outputs go to ./outputs (figures, reports, trained model).
"""
import json

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config as cfg
from src import attendance_analysis, eda, modeling, recommendation, risk_labeling
from src.preprocessing import prepare_data


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main():
    for d in (cfg.FIG_DIR, cfg.REPORT_DIR, cfg.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1 & 2. Load, clean, engineer ----------------------------------
    banner("STEP 2 - DATA CLEANING")
    df, log = prepare_data()
    print(json.dumps(log, indent=2))
    modeling.save_json(log, "cleaning_log.json")
    df.to_csv(cfg.ROOT / "data" / "cleaned_student_data.csv", index=False)

    # ---- 1. Exploration --------------------------------------------------
    banner("STEP 1 - DATA EXPLORATION")
    eda.run_eda(df)

    # ---- 1b. Focus: attendance vs final mark ----------------------------
    banner("STEP 1b - ATTENDANCE vs FINAL MARK (focused analysis)")
    attendance_analysis.run_attendance_analysis(df)

    # ---- 3a. Risk labels -------------------------------------------------
    banner("STEP 3 - RISK LABELS (rule-based criteria)")
    df = risk_labeling.assign_risk(df)
    dist = df["Risk"].value_counts().reindex(cfg.RISK_ORDER)
    print(pd.DataFrame({"students": dist, "share": (dist / len(df)).round(3)}))
    eda.plot_risk_overview(df)

    # ---- 3b. Feature selection + model comparison -----------------------
    banner("STEP 3 - FEATURE SELECTION & MODEL COMPARISON")
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    ranking = modeling.select_features(X_train, y_train)
    selected = ranking[ranking >= ranking.mean()].index.tolist()
    print("Importance ranking (training split):\n", ranking.round(3).to_string())
    print(f"\nSelected {len(selected)} features (importance >= mean): {selected}")

    models = modeling.build_models()
    comparison = modeling.compare_models(models, X_train[selected], y_train)
    print("\n5-fold CV on training data:\n", comparison.round(3).to_string(index=False))
    comparison.to_csv(cfg.REPORT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.loc[comparison["Model"] != "Baseline (majority class)"].iloc[0]["Model"]
    best = models[best_name].fit(X_train[selected], y_train)

    # ---- 4. Evaluation on the held-out test set -------------------------
    banner(f"STEP 4 - EVALUATION OF '{best_name}' ON HELD-OUT TEST SET")
    result, text = modeling.evaluate(best, X_test[selected], y_test, best_name)
    print(text)
    print(f"Accuracy={result['accuracy']:.3f}  Macro-Precision={result['precision_macro']:.3f}  "
          f"Macro-Recall={result['recall_macro']:.3f}  Macro-F1={result['f1_macro']:.3f}")
    print("Confusion matrix (rows=actual, cols=predicted; order Low/Medium/High):")
    print(pd.DataFrame(result["confusion_matrix"], index=cfg.RISK_ORDER, columns=cfg.RISK_ORDER))
    importance = modeling.plot_importance(best, X_test[selected], y_test, best_name)
    print("\nPermutation importance:\n", importance.round(4).to_string())

    # ---- Honest sanity check on a real outcome --------------------------
    banner("SANITY CHECK - CAN WE PREDICT ACTUAL FAILURE (GRADE D/F)?")
    check = modeling.outcome_only_check(df)
    print(json.dumps(check, indent=2))
    modeling.save_json({"selected_features": selected, "best_model": best_name, "test_metrics": result,
                        "cv_comparison": comparison.to_dict("records"), "outcome_check": check},
                       "metrics.json")
    (cfg.REPORT_DIR / "classification_report.txt").write_text(text, encoding="utf-8")

    # ---- 5. Prediction output for sample students -----------------------
    banner("STEP 5 - SAMPLE PREDICTIONS")
    # Deployment model: refit on ALL students (the test metrics above remain the honest estimate).
    final_model = models[best_name].fit(df[selected], df["Risk"].astype(str))
    joblib.dump({"model": final_model, "features": selected}, cfg.MODEL_PATH)
    df["Predicted_Risk"] = final_model.predict(df[selected])
    for level in ["High", "Medium", "Low"]:
        row = df[df["Predicted_Risk"] == level].iloc[0]
        print(recommendation.format_prediction(row, level), "\n" + "-" * 40)

    # ---- 6. Automated report (out-of-fold predictions = no student is scored by a model that saw them)
    banner("STEP 6 - AUTOMATED REPORT")
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE)
    oof = cross_val_predict(models[best_name], df[selected], df["Risk"].astype(str), cv=cv, n_jobs=-1)
    report = recommendation.generate_report(df, pd.Series(oof, index=df.index))
    recommendation.save_report(report)
    print(report.head(8)[["Student_ID", "Attendance", "Risk", "Recommendation"]].to_string(index=False))
    print(f"\nSaved full report ({len(report):,} students) to {cfg.REPORT_DIR}")


if __name__ == "__main__":
    main()
