"""
Step 1 - Data exploration.

Goal: understand the data and, specifically, test the relationship between
attendance and marks. Every figure is saved to outputs/figures and every
statistic to outputs/reports/eda_summary.txt so the findings are reproducible.
"""
import matplotlib
matplotlib.use("Agg")  # no display needed - write straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as cfg

sns.set_theme(style="whitegrid", context="notebook")
GRADE_ORDER = ["A", "B", "C", "D", "F"]
BAND_LABELS = ["<60%", "60-75%", "75-90%", "90-100%"]
# explicit colours so "green = good, red = bad" is never flipped by a palette's ordering
GOOD_TO_BAD = ["#55A868", "#9BC46B", "#F2D16B", "#E8955A", "#C44E52"]
BAD_TO_GOOD = ["#C44E52", "#E8955A", "#F2D16B", "#55A868"]


def _save(fig, name):
    path = cfg.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame) -> dict:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    lines = []

    df = df.copy()
    df["Grade_Points"] = df["Grade"].map({"A": 4, "B": 3, "C": 2, "D": 1, "F": 0})
    df["Attendance_Band"] = pd.cut(df["Attendance"], [-0.1, 60, 75, 90, 100], labels=BAND_LABELS)

    # ---- Fig 1: distributions -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, colr in zip(axes.ravel(), ["Attendance", "Marks", "Total_Score", "Participation_Score"],
                             ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color=colr)
        ax.set_title(f"Distribution of {col}")
    _save(fig, "01_distributions.png")

    # ---- Attendance vs marks: the key question ---------------------------
    r_m, p_m = stats.pearsonr(df["Attendance"], df["Marks"])
    rho_m, _ = stats.spearmanr(df["Attendance"], df["Marks"])
    r_t, p_t = stats.pearsonr(df["Attendance"], df["Total_Score"])
    rho_t, _ = stats.spearmanr(df["Attendance"], df["Total_Score"])
    lines += [
        "ATTENDANCE vs MARKS",
        f"  Attendance vs Marks (weighted course-work): Pearson r={r_m:+.3f} (p={p_m:.3f}), Spearman rho={rho_m:+.3f}",
        f"  Attendance vs Total_Score:                  Pearson r={r_t:+.3f} (p={p_t:.3f}), Spearman rho={rho_t:+.3f}",
    ]
    out.update(r_att_marks=r_m, r_att_total=r_t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, r in zip(axes, ["Marks", "Total_Score"], [r_m, r_t]):
        ax.scatter(df["Attendance"], df[col], s=6, alpha=0.25, color="#4C72B0")
        slope, intercept = np.polyfit(df["Attendance"], df[col], 1)
        xs = np.linspace(df["Attendance"].min(), df["Attendance"].max(), 50)
        ax.plot(xs, slope * xs + intercept, color="crimson", lw=2, label=f"trend (r = {r:+.2f})")
        ax.set_xlabel("Attendance (%)")
        ax.set_ylabel(col)
        ax.set_title(f"Attendance vs {col}")
        ax.legend()
    _save(fig, "02_attendance_vs_marks.png")

    # ---- Attendance vs Grade: where the real signal is -------------------
    h, p_kw = stats.kruskal(*[g["Attendance"].values for _, g in df.groupby("Grade")])
    att_by_grade = df.groupby("Grade")["Attendance"].mean().reindex(GRADE_ORDER)
    fail_rate = (df["Grade"].isin(["D", "F"])).groupby(df["Attendance_Band"], observed=True).mean()
    r_g, _ = stats.spearmanr(df["Attendance"], df["Grade_Points"])
    lines += [
        "",
        "ATTENDANCE vs GRADE",
        f"  Spearman rho (Attendance, Grade points) = {r_g:+.3f}; Kruskal-Wallis across grades p = {p_kw:.2e}",
        "  Mean attendance by grade: " + ", ".join(f"{g}={v:.1f}%" for g, v in att_by_grade.items()),
        "  Share of students with grade D/F by attendance band: "
        + ", ".join(f"{b}={v:.1%}" for b, v in fail_rate.items()),
    ]
    out.update(rho_att_grade=r_g, fail_rate_by_band=fail_rate.to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="Grade", y="Attendance", order=GRADE_ORDER, hue="Grade", hue_order=GRADE_ORDER,
                legend=False, palette=dict(zip(GRADE_ORDER, GOOD_TO_BAD)), ax=axes[0])
    axes[0].set_title("Attendance by final grade")
    sns.barplot(x=fail_rate.index.astype(str), y=fail_rate.values * 100, hue=fail_rate.index.astype(str),
                legend=False, palette=dict(zip(BAND_LABELS, BAD_TO_GOOD)), ax=axes[1])
    axes[1].set_xlabel("Attendance band")
    axes[1].set_ylabel("% of students with grade D or F")
    axes[1].set_title("Low attendance -> more D/F grades")
    for i, v in enumerate(fail_rate.values * 100):
        axes[1].text(i, v + 0.5, f"{v:.0f}%", ha="center")
    _save(fig, "03_attendance_vs_grade.png")

    # ---- Correlation heatmap --------------------------------------------
    num = ["Age", "Attendance", "Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
           "Participation_Score", "Projects_Score", "Marks", "Total_Score", "Grade_Points",
           "test_preparation_course", "math_score", "reading_score", "writing_score", "science_score"]
    corr = df[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Spearman correlation matrix")
    _save(fig, "04_correlation_heatmap.png")

    # ---- Other patterns worth reporting ---------------------------------
    strongest = (corr["Grade_Points"].drop("Grade_Points").abs().sort_values(ascending=False).head(3))
    r_tot_marks, _ = stats.spearmanr(df["Total_Score"], df["Marks"])
    prep = df.groupby("test_preparation_course")[["Marks", "Total_Score"]].mean().round(2)
    dept = df.groupby("Department")[["Attendance", "Marks"]].mean().round(2)
    gender = df.groupby("Gender")[["Attendance", "Marks"]].mean().round(2)
    lines += [
        "",
        "OTHER PATTERNS",
        "  Features most correlated with Grade (abs Spearman): "
        + ", ".join(f"{k}={corr.loc[k, 'Grade_Points']:+.3f}" for k in strongest.index),
        f"  Total_Score vs weighted Marks: Spearman rho = {r_tot_marks:+.3f}  -> Total_Score is NOT explained by the components",
        f"  Total_Score by Grade (means): " + ", ".join(
            f"{g}={v:.1f}" for g, v in df.groupby('Grade')['Total_Score'].mean().reindex(GRADE_ORDER).items()),
        "  Test-prep course (0=no, 1=yes):\n" + prep.to_string().replace("\n", "\n    "),
        "  By department:\n" + dept.to_string().replace("\n", "\n    "),
        "  By gender:\n" + gender.to_string().replace("\n", "\n    "),
    ]
    out["total_vs_marks_rho"] = r_tot_marks

    (cfg.REPORT_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out


def plot_risk_overview(df: pd.DataFrame):
    """Visualise the risk labels produced by the rules (called after labelling)."""
    palette = {"Low": "#55A868", "Medium": "#DD8452", "High": "#C44E52"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    counts = df["Risk"].value_counts().reindex(cfg.RISK_ORDER)
    axes[0].bar(counts.index, counts.values, color=[palette[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 40, f"{v:,}\n({v / len(df):.0%})", ha="center")
    axes[0].set_title("Students per risk level")
    axes[0].set_ylim(0, counts.max() * 1.18)

    sns.scatterplot(data=df.sample(3000, random_state=cfg.RANDOM_STATE), x="Attendance", y="Marks",
                    hue="Risk", hue_order=cfg.RISK_ORDER, palette=palette, s=14, alpha=0.6, ax=axes[1])
    axes[1].axvline(cfg.ATTENDANCE_WARN, ls="--", color="grey", lw=1)
    axes[1].axhline(cfg.MARKS_WARN, ls="--", color="grey", lw=1)
    axes[1].set_title("Attendance vs Marks, coloured by risk")

    ct = pd.crosstab(df["Department"], df["Risk"], normalize="index")[cfg.RISK_ORDER] * 100
    ct.plot(kind="barh", stacked=True, color=[palette[k] for k in cfg.RISK_ORDER], ax=axes[2])
    axes[2].set_xlabel("% of department")
    axes[2].set_title("Risk mix by department")
    axes[2].legend(title="Risk", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return _save(fig, "05_risk_overview.png")
