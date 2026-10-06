"""
Step 2 - Data preprocessing.

What is wrong with the raw file (found during exploration):
  * 30 exact duplicate rows (duplicated Student_IDs)
  * ~200 missing values in each of 8 numeric columns
  * impossible values: attendance > 100 % or < 0, scores > 100 or < 0,
    ages of -3 and 87
  * inconsistent text: ' FEMALE', ' MALE', ' engineering', 'BUSINESS',
    and the same department written as 'CS' / 'Computer Science' / 'Math' / 'Mathematics'
  * math_score is stored as text because one cell contains a tab ('\\t41')
"""
import numpy as np
import pandas as pd

from . import config as cfg


def load_raw(path=cfg.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(raw: pd.DataFrame):
    """Return (clean_df, log). `log` records what was fixed so it can be reported."""
    df = raw.copy()
    log = {"rows_raw": len(df)}

    # 1. Tidy column names: "Attendance (%)" -> "Attendance"
    df.columns = [c.replace(" (%)", "").strip() for c in df.columns]

    # 2. Fix text columns: strip whitespace, unify case, merge department aliases
    for col in ["Gender", "Department", "Grade"]:
        df[col] = df[col].astype(str).str.strip()
    df["Gender"] = df["Gender"].str.title()
    df["Grade"] = df["Grade"].str.upper()
    df["Department"] = df["Department"].str.lower().map(cfg.DEPARTMENT_MAP)

    # 3. math_score arrived as text because of a stray tab -> coerce to number
    df["math_score"] = pd.to_numeric(df["math_score"].astype(str).str.strip(), errors="coerce")

    # 4. Remove duplicate students (keep the first record)
    before = len(df)
    df = df.drop_duplicates(subset="Student_ID", keep="first").reset_index(drop=True)
    log["duplicates_removed"] = before - len(df)

    # 5. Impossible values -> NaN (they are errors, not extreme-but-real students)
    invalid = {}
    for col, (lo, hi) in cfg.VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    log["invalid_values_set_to_nan"] = {k: v for k, v in invalid.items() if v}

    # 6. Impute missing values. Median is robust to the remaining skew.
    #    (< 2.1 % of any column is missing, so imputation has a negligible effect;
    #    dropping rows would have thrown away ~15 % of students.)
    missing = df.isna().sum()
    log["missing_before_imputation"] = {k: int(v) for k, v in missing.items() if v}
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["Department"] = df["Department"].fillna(df["Department"].mode()[0])
    log["rows_clean"] = len(df)
    return df, log


def add_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: weighted course-work average on a 0-100 scale."""
    df = df.copy()
    df["Marks"] = sum(df[c] * w for c, w in cfg.MARK_WEIGHTS.items())
    return df


def prepare_data(path=cfg.DATA_PATH):
    """Convenience wrapper used by main.py and predict_student.py: load -> clean -> engineer Marks."""
    clean, log = clean_data(load_raw(path))
    return add_marks(clean), log
