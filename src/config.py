"""
Central configuration: paths, valid ranges, and the risk-scoring rules.

Keeping every "magic number" here means the analytical choices (what counts
as low attendance? how much does the final exam weigh?) are visible in one
place and can be changed without touching the pipeline code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dcs_student_data.csv"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Data-cleaning rules
# --------------------------------------------------------------------------
# Values outside these ranges are physically impossible (e.g. attendance of
# 135 %, age of -3) so they are treated as data-entry errors -> set to NaN
# and imputed later.
VALID_RANGES = {
    "Age": (16, 40),                 # plausible university-student ages
    "Attendance": (0, 100),
    "Midterm_Score": (0, 100),
    "Final_Score": (0, 100),
    "Assignments_Avg": (0, 100),
    "Quizzes_Avg": (0, 100),
    "Participation_Score": (0, 10),
    "Projects_Score": (0, 100),
    "Total_Score": (0, 100),
    "math_score": (0, 100),
    "reading_score": (0, 100),
    "writing_score": (0, 100),
    "science_score": (0, 100),
}

# The same department appears under several spellings / cases.
DEPARTMENT_MAP = {
    "cs": "Computer Science",
    "computer science": "Computer Science",
    "engineering": "Engineering",
    "business": "Business",
    "math": "Mathematics",
    "mathematics": "Mathematics",
}

# --------------------------------------------------------------------------
# "Marks" = weighted average of the course-work components.
# Weights follow a typical university scheme (finals count the most).
# Participation is on a 0-10 scale and is tracked separately.
# --------------------------------------------------------------------------
MARK_WEIGHTS = {
    "Midterm_Score": 0.25,
    "Final_Score": 0.30,
    "Assignments_Avg": 0.15,
    "Quizzes_Avg": 0.10,
    "Projects_Score": 0.20,
}
COMPONENT_LABELS = {
    "Midterm_Score": "the midterm material",
    "Final_Score": "final-exam preparation",
    "Assignments_Avg": "assignments",
    "Quizzes_Avg": "quizzes",
    "Projects_Score": "projects",
}

# --------------------------------------------------------------------------
# Risk-scoring rules (my own criteria - see README for the reasoning)
# Each indicator contributes 0, 1 or 2 "risk points"; total range is 0-6.
# --------------------------------------------------------------------------
ATTENDANCE_SEVERE = 60    # below this -> 2 points  (well under typical 75 % rule)
ATTENDANCE_WARN = 75      # below this -> 1 point   (common minimum-attendance rule)
MARKS_SEVERE = 60         # below this -> 2 points  (failing / near-failing)
MARKS_WARN = 70           # below this -> 1 point   (below a solid pass)
GRADE_POINTS = {"A": 0, "B": 0, "C": 1, "D": 2, "F": 2}

HIGH_RISK_MIN_POINTS = 4  # 4-6 points -> High
MEDIUM_RISK_MIN_POINTS = 2  # 2-3 points -> Medium ; 0-1 -> Low
RISK_ORDER = ["Low", "Medium", "High"]
