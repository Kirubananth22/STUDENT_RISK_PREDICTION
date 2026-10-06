"""
Focused analysis: how does ATTENDANCE relate to the FINAL MARK?

"Final mark" can mean three things in this dataset, so all three are tested:
  1. Final_Score  - the final-exam score (0-100)
  2. Total_Score  - the overall course score (0-100)
  3. Grade        - the letter grade (A-F)

For each one we ask: is there a relationship, how strong is it, and is it
statistically significant? Results go to outputs/figures/10_*.png ... and
outputs/reports/attendance_analysis.txt.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from . import config as cfg

sns.set_theme(style="whitegrid", context="notebook")
GRADES = ["A", "B", "C", "D", "F"]
GRADE_COLORS = ["#55A868", "#9BC46B", "#F2D16B", "#E8955A", "#C44E52"]
BINS = [49.9, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100.1]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / name, dpi=130)
    plt.close(fig)


def _binned_mean(x, y):
    """Mean of y (with 95 % CI) inside attendance bins - reveals the trend the scatter hides."""
    b = pd.cut(x, BINS)
    g = y.groupby(b, observed=True)
    mean, sem = g.mean(), g.sem()
    mid = np.array([iv.mid for iv in mean.index])
    return mid, mean.values, 1.96 * sem.values


def run_attendance_analysis(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["Grade_Points"] = df["Grade"].map({"A": 4, "B": 3, "C": 2, "D": 1, "F": 0})
    att = df["Attendance"]
    lines, out = [], {}

    # ------------------------------------------------------------------ stats
    lines.append("1) ATTENDANCE vs FINAL-EXAM SCORE and TOTAL SCORE (numeric)")
    for col in ["Final_Score", "Total_Score", "Marks"]:
        r, p = stats.pearsonr(att, df[col])
        rho, p_s = stats.spearmanr(att, df[col])
        slope = np.polyfit(att, df[col], 1)[0]
        lines.append(f"   {col:12s} Pearson r={r:+.3f} (p={p:.3f})  Spearman rho={rho:+.3f}  "
                     f"slope={slope:+.3f} points per +1% attendance  R^2={r*r:.4f}")
        out[col] = dict(r=r, p=p, rho=rho, slope=slope)

    low, high = df[att < cfg.ATTENDANCE_WARN], df[att >= cfg.ATTENDANCE_WARN]
    t, p_t = stats.ttest_ind(low["Final_Score"], high["Final_Score"], equal_var=False)
    lines.append(f"\n   Final_Score: attendance <75% mean={low['Final_Score'].mean():.2f} "
                 f"vs >=75% mean={high['Final_Score'].mean():.2f}  (Welch t-test p={p_t:.3f})")
    t2, p_t2 = stats.ttest_ind(low["Total_Score"], high["Total_Score"], equal_var=False)
    lines.append(f"   Total_Score: attendance <75% mean={low['Total_Score'].mean():.2f} "
                 f"vs >=75% mean={high['Total_Score'].mean():.2f}  (Welch t-test p={p_t2:.3f})")

    lines.append("\n2) ATTENDANCE vs LETTER GRADE (categorical)")
    rho_g, p_g = stats.spearmanr(att, df["Grade_Points"])
    h, p_kw = stats.kruskal(*[df.loc[df.Grade == g, "Attendance"] for g in GRADES])
    lines.append(f"   Spearman rho (attendance, grade points) = {rho_g:+.3f} (p={p_g:.2e})")
    lines.append(f"   Kruskal-Wallis (attendance differs across grades) p = {p_kw:.2e}")
    lines.append("   Mean attendance by grade: " + ", ".join(
        f"{g}={df.loc[df.Grade == g, 'Attendance'].mean():.1f}%" for g in GRADES))
    out["grade"] = dict(rho=rho_g, p=p_g)

    # grade mix by attendance band
    df["Band"] = pd.cut(att, [0, 60, 75, 90, 100], labels=["<60%", "60-75%", "75-90%", "90-100%"], include_lowest=True)
    mix = pd.crosstab(df["Band"], df["Grade"], normalize="index")[GRADES] * 100
    lines.append("\n   Grade mix (% of students) by attendance band:\n" + mix.round(1).to_string().replace("\n", "\n   "))
    chi2, p_chi, _, _ = stats.chi2_contingency(pd.crosstab(df["Band"], df["Grade"]))
    lines.append(f"   Chi-square test (band vs grade): chi2={chi2:.1f}, p={p_chi:.2e}")
    # effect size: odds of A/B for 90-100 vs <60
    ab = df["Grade"].isin(["A", "B"]).groupby(df["Band"], observed=True).mean() * 100
    lines.append("   % of students earning A or B: " + ", ".join(f"{b}={v:.0f}%" for b, v in ab.items()))
    out["ab_by_band"] = ab.to_dict()

    # ------------------------------------------------------------------ figures
    # Fig 10: numeric final marks - the scatter plus the binned trend (shows "no relationship")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for j, col in enumerate(["Final_Score", "Total_Score"]):
        ax = axes[0, j]
        hb = ax.hexbin(att, df[col], gridsize=28, cmap="Blues", mincnt=1)
        slope, icpt = np.polyfit(att, df[col], 1)
        xs = np.array([50, 100])
        ax.plot(xs, slope * xs + icpt, color="crimson", lw=2.5,
                label=f"linear fit: r = {out[col]['r']:+.3f}, p = {out[col]['p']:.2f}")
        ax.set(xlabel="Attendance (%)", ylabel=col, title=f"Attendance vs {col}  (density of 10,000 students)")
        ax.legend(loc="lower right")
        fig.colorbar(hb, ax=ax, label="students")
        ax = axes[1, j]
        mid, m, ci = _binned_mean(att, df[col])
        ax.errorbar(mid, m, yerr=ci, fmt="o-", color="#4C72B0", capsize=4, lw=2)
        ax.axhline(df[col].mean(), color="grey", ls="--", label=f"overall mean = {df[col].mean():.1f}")
        ax.set(xlabel="Attendance (%) - binned", ylabel=f"Mean {col} (95% CI)",
               title=f"Average {col} at each attendance level", ylim=(df[col].mean() - 6, df[col].mean() + 6))
        ax.legend()
    _save(fig, "10_attendance_vs_final_score.png")

    # Fig 11: the relationship that IS there - attendance vs letter grade
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    sns.boxplot(data=df, x="Grade", y="Attendance", order=GRADES, hue="Grade", hue_order=GRADES,
                palette=dict(zip(GRADES, GRADE_COLORS)), legend=False, ax=axes[0])
    means = df.groupby("Grade")["Attendance"].mean().reindex(GRADES)
    axes[0].scatter(range(5), means.values, color="black", marker="D", zorder=5, label="mean")
    for i, v in enumerate(means.values):
        axes[0].text(i + 0.08, v + 1.2, f"{v:.0f}%", fontsize=9)
    axes[0].set(title="Attendance distribution for each final grade", ylabel="Attendance (%)")
    axes[0].legend(loc="lower left")

    mix.plot(kind="bar", stacked=True, color=GRADE_COLORS, ax=axes[1], width=0.75, edgecolor="white")
    axes[1].set(title="Grade mix by attendance band", xlabel="Attendance band", ylabel="% of students")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Grade", bbox_to_anchor=(1.01, 1), loc="upper left")
    for i, band in enumerate(mix.index):
        axes[1].text(i, 50, "", ha="center")

    # smooth probability curves of good / poor outcomes across attendance
    for label, grades, colr in [("Earned A or B", ["A", "B"], "#55A868"), ("Got D or F", ["D", "F"], "#C44E52")]:
        flag = df["Grade"].isin(grades).astype(float)
        mid, m, ci = _binned_mean(att, flag * 100)
        axes[2].plot(mid, m, "o-", color=colr, lw=2.5, label=label)
        axes[2].fill_between(mid, m - ci, m + ci, color=colr, alpha=0.18)
    axes[2].set(title="Chance of a good / poor grade vs attendance", xlabel="Attendance (%)",
                ylabel="% of students (95% CI)")
    axes[2].legend()
    _save(fig, "11_attendance_vs_grade_detail.png")

    # Fig 12: one-picture summary - every "final mark" definition, standardised
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = ["Final_Score", "Total_Score", "Marks", "Grade (A=4..F=0)"]
    vals = [out["Final_Score"]["rho"], out["Total_Score"]["rho"], out["Marks"]["rho"], rho_g]
    cols = ["#9aa5b1" if abs(v) < 0.1 else "#4C72B0" for v in vals]
    bars = ax.barh(labels, vals, color=cols)
    ax.axvline(0, color="black", lw=1)
    for b, v in zip(bars, vals):
        ax.text(v + (0.008 if v >= 0 else -0.008), b.get_y() + b.get_height() / 2, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontweight="bold")
    ax.set(xlim=(-0.1, 0.4), xlabel="Spearman correlation with attendance",
           title="Attendance vs every version of the 'final mark'")
    _save(fig, "12_correlation_summary.png")

    # ---- Step-change finding: outcomes jump at ~60 % and ~85 % attendance --------------
    # The binned curves (fig 11, right) are flat inside tiers and jump between them, so the
    # relationship is a THRESHOLD effect, not a straight line - which is why Pearson r (linear)
    # on the numeric scores is ~0 while the grade relationship is highly significant.
    # Rows whose attendance was median-imputed (a spike at the median) are excluded here.
    real = df[att != att.median()].copy()
    # best single split point for P(A or B), found by maximising variance reduction
    real = real.sort_values("Attendance")
    y, a = real["Grade"].isin(["A", "B"]).astype(float).values, real["Attendance"].values
    gain = [(np.var(y) * len(y) - np.var(y[:i]) * i - np.var(y[i:]) * (len(y) - i), a[i])
            for i in range(500, len(y) - 500, 20)]
    best_split = max(gain)[1]
    real["Tier"] = pd.cut(real["Attendance"], [0, 60, 85, 100], labels=["<60%", "60-85%", "85-100%"], right=False)
    tier = pd.DataFrame({
        "students": real.groupby("Tier", observed=True).size(),
        "pct_A_or_B": real["Grade"].isin(["A", "B"]).groupby(real["Tier"], observed=True).mean() * 100,
        "pct_D_or_F": real["Grade"].isin(["D", "F"]).groupby(real["Tier"], observed=True).mean() * 100,
    })
    lines.append(f"\n3) STEP-CHANGE (threshold) EFFECT. Best single split point for P(A or B): attendance = {best_split:.0f}%")
    lines.append("   Outcome by attendance tier (median-imputed rows excluded):\n   " + tier.round(1).to_string().replace("\n", "\n   "))
    out["tiers"] = tier.to_dict()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = np.arange(len(tier)); w = 0.36
    b1 = ax.bar(xs - w / 2, tier["pct_A_or_B"], w, color="#55A868", label="Earned A or B")
    b2 = ax.bar(xs + w / 2, tier["pct_D_or_F"], w, color="#C44E52", label="Got D or F")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}%", ha="center", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"{t}\n(n={n:,})" for t, n in zip(tier.index, tier["students"])])
    ax.set(ylabel="% of students", ylim=(0, 85), title="Three attendance tiers, three very different outcomes")
    ax.legend()
    _save(fig, "13_attendance_tiers.png")

    (cfg.REPORT_DIR / "attendance_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out
