"""
Steps 5 & 6 - Prediction output, recommendations and the automatic report.

The MODEL decides the risk level; the recommendation text is then built from the
student's own numbers, so advice is specific ("quizzes are your weakest area")
rather than generic.
"""
import pandas as pd

from . import config as cfg


def build_recommendation(row: pd.Series, risk: str) -> str:
    att, marks = row["Attendance"], row["Marks"]
    weakest = min(cfg.MARK_WEIGHTS, key=lambda c: row[c])
    weak_tip = (f"Give extra practice to {cfg.COMPONENT_LABELS[weakest]} (current average {row[weakest]:.0f})."
                if row[weakest] < 65 else None)

    # Low risk: encourage; only mention a weak component as an optional tip.
    if risk == "Low":
        msg = "Keep up the good work and maintain current attendance and assessment performance."
        return msg + (" Optional tip: " + weak_tip[0].lower() + weak_tip[1:] if weak_tip else "")

    parts = []
    # Main action: whatever is below target (attendance first - it is the strongest driver)
    focus = []
    if att < cfg.ATTENDANCE_WARN:
        focus.append("improve attendance")
    if marks < cfg.MARKS_WARN:
        focus.append("focus on upcoming assessments")
    if focus:
        parts.append(" and ".join(focus).capitalize() + ".")
    else:
        parts.append("No single weak area stands out - review overall study habits with an adviser.")

    # Specific advice from the student's own numbers
    if weak_tip:
        parts.append(weak_tip)
    if row["Participation_Score"] < 3:
        parts.append("Take part more actively in class discussions.")

    # Escalation depends on the model's risk level
    parts.append("Meet an academic adviser this week to agree a support plan." if risk == "High"
                 else "Check progress with the course mentor after the next assessment.")
    return " ".join(parts)


def format_prediction(row: pd.Series, risk: str) -> str:
    """Console output in the format requested in the brief."""
    return (f"Student: {row['Student_ID']} ({row['First_Name']} {row['Last_Name']})\n"
            f"Attendance: {row['Attendance']:.0f}%\n"
            f"Marks: {row['Marks']:.0f}\n"
            f"Risk: {str(risk).upper()}\n\n"
            f"Recommendation:\n{build_recommendation(row, risk)}")


def generate_report(df: pd.DataFrame, risk: pd.Series) -> pd.DataFrame:
    """Student | Attendance | Risk | Recommendation for every student, highest risk first."""
    rep = pd.DataFrame({
        "Student_ID": df["Student_ID"],
        "Student": df["First_Name"] + " " + df["Last_Name"],
        "Department": df["Department"],
        "Attendance": df["Attendance"].round(1),
        "Marks": df["Marks"].round(1),
        "Risk": pd.Categorical(risk.values, categories=cfg.RISK_ORDER, ordered=True),
    })
    rep["Recommendation"] = [build_recommendation(r, k) for (_, r), k in zip(df.iterrows(), risk)]
    rep = rep.sort_values(["Risk", "Attendance"], ascending=[False, True]).reset_index(drop=True)
    return rep


def save_report(rep: pd.DataFrame):
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(cfg.REPORT_DIR / "student_risk_report.csv", index=False)
    rep[rep["Risk"] == "High"].to_csv(cfg.REPORT_DIR / "high_risk_students.csv", index=False)

    colors = {"High": "#f8d7da", "Medium": "#fff3cd", "Low": "#d4edda"}
    rows = "\n".join(
        f"<tr style='background:{colors[r.Risk]}'><td>{r.Student_ID}</td><td>{r.Student}</td>"
        f"<td>{r.Attendance:.0f}%</td><td>{r.Risk}</td><td>{r.Recommendation}</td></tr>"
        for r in rep.head(200).itertuples())
    counts = rep["Risk"].value_counts().reindex(cfg.RISK_ORDER)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Student Risk Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:6px 10px;text-align:left;font-size:14px}}th{{background:#333;color:#fff}}</style></head><body>
<h1>Student Academic-Risk Report</h1>
<p>{len(rep):,} students &mdash; High: {counts['High']:,} &nbsp; Medium: {counts['Medium']:,} &nbsp; Low: {counts['Low']:,}.
Showing the 200 highest-priority students; the full list is in <code>student_risk_report.csv</code>.</p>
<table><tr><th>Student</th><th>Name</th><th>Attendance</th><th>Risk</th><th>Recommendation</th></tr>
{rows}</table></body></html>"""
    (cfg.REPORT_DIR / "student_risk_report.html").write_text(html, encoding="utf-8")
