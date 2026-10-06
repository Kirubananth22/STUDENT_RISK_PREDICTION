"""
Look up one student and print the risk prediction + recommendation.

Usage:  python predict_student.py S1000
(run `python main.py` first so the trained model exists)
"""
import sys

import joblib

from src import config as cfg
from src.preprocessing import prepare_data
from src.recommendation import format_prediction


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict_student.py <Student_ID>   e.g. S1000")
    if not cfg.MODEL_PATH.exists():
        sys.exit("No trained model found - run `python main.py` first.")
    df, _ = prepare_data()
    match = df[df["Student_ID"] == sys.argv[1].strip().upper()]
    if match.empty:
        sys.exit(f"Student {sys.argv[1]} not found.")
    bundle = joblib.load(cfg.MODEL_PATH)
    row = match.iloc[0]
    risk = bundle["model"].predict(match[bundle["features"]])[0]
    print(format_prediction(row, risk))


if __name__ == "__main__":
    main()
