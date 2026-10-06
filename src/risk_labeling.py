"""
Step 3a - Defining Low / Medium / High risk.

The dataset has no "at risk" column, so I define one. The approach is a
transparent points system - easy to explain to teachers and advisers, and each
point can be traced to a concrete reason.

Three indicators, each worth 0 / 1 / 2 points:

  Indicator   | 0 pts      | 1 pt        | 2 pts
  ------------|------------|-------------|----------
  Attendance  | >= 75 %    | 60 - 75 %   | < 60 %
  Marks       | >= 70      | 60 - 70     | < 60
  Final Grade | A or B     | C           | D or F

  Total 0-1 -> LOW,   2-3 -> MEDIUM,   4-6 -> HIGH

Why these indicators?
  * Attendance - the strongest predictor of grade in this dataset (see EDA) and
    something an adviser can act on immediately.
  * Marks      - weighted course-work average: what the student has actually achieved.
  * Grade      - the official outcome; it captures failure that the other two miss.

Why not simply use Total_Score? EDA showed it is statistically unrelated to every
other column (R^2 ~ 0.001 against the components, and flat across grades), so
using it would add noise rather than information.

IMPORTANT for the ML step: the *label* uses Grade, but the *model* is never shown
Grade (or Total_Score). The model has to predict risk from early-warning
information (attendance + assessment scores) - which is the real-world use case,
because a grade does not exist until the semester is over.
"""
import numpy as np
import pandas as pd

from . import config as cfg


def attendance_points(att: pd.Series) -> pd.Series:
    return np.select([att < cfg.ATTENDANCE_SEVERE, att < cfg.ATTENDANCE_WARN], [2, 1], default=0)


def marks_points(marks: pd.Series) -> pd.Series:
    return np.select([marks < cfg.MARKS_SEVERE, marks < cfg.MARKS_WARN], [2, 1], default=0)


def assign_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Attendance_Pts"] = attendance_points(df["Attendance"])
    df["Marks_Pts"] = marks_points(df["Marks"])
    df["Grade_Pts"] = df["Grade"].map(cfg.GRADE_POINTS)
    df["Risk_Points"] = df["Attendance_Pts"] + df["Marks_Pts"] + df["Grade_Pts"]
    df["Risk"] = np.select(
        [df["Risk_Points"] >= cfg.HIGH_RISK_MIN_POINTS, df["Risk_Points"] >= cfg.MEDIUM_RISK_MIN_POINTS],
        ["High", "Medium"], default="Low")
    df["Risk"] = pd.Categorical(df["Risk"], categories=cfg.RISK_ORDER, ordered=True)
    return df
