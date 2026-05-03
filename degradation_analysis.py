"""
degradation_analysis.py
Computes per-model degradation rates across distortion levels and
produces a robustness summary with ranking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_degradation(rel_df, metric="accuracy"):
    """
    Compute degradation stats for each model across distortion levels.

    Parameters
    ----------
    rel_df  : pd.DataFrame
        Long-form results with columns:
        model_name, distortion_level, accuracy, precision, recall, f1, roc_auc, confidence
        (same format built in app.py from all_results)
    metric  : str
        Which metric to analyse (default: accuracy)

    Returns
    -------
    summary_df : pd.DataFrame
        Per-model: Baseline, Worst, Abs Drop, % Drop, Deg Rate/Level, Robustness Score, Rank
    """
    rows = []

    for model in rel_df["model_name"].unique():
        mdf    = rel_df[rel_df["model_name"] == model].sort_values("distortion_level")
        levels = mdf["distortion_level"].values
        values = mdf[metric].values

        if len(values) < 2:
            continue

        baseline  = values[0]
        worst     = values[-1]
        abs_drop  = baseline - worst
        pct_drop  = (abs_drop / (baseline + 1e-9)) * 100

        # Degradation rate: accuracy lost per unit of distortion
        level_range = levels[-1] - levels[0]
        deg_rate    = abs_drop / (level_range + 1e-9)

        # Robustness score: area under performance curve normalised to 0-100
        area              = np.trapezoid(values, levels)
        max_possible_area = baseline * level_range
        robustness_score  = (area / (max_possible_area + 1e-9)) * 100 if max_possible_area > 0 else 100.0
        robustness_score  = min(robustness_score, 100.0)

        rows.append({
            "Model":           model,
            "Baseline":        round(float(baseline), 4),
            "Worst":           round(float(worst),    4),
            "Abs Drop":        round(float(abs_drop), 4),
            "% Drop":          round(float(pct_drop), 2),
            "Deg Rate/Level":  round(float(deg_rate), 4),
            "Robustness Score":round(float(robustness_score), 2),
        })

    summary_df        = pd.DataFrame(rows).sort_values("Robustness Score", ascending=False)
    summary_df["Rank"]= range(1, len(summary_df) + 1)
    cols = ["Rank", "Model", "Baseline", "Worst", "Abs Drop",
            "% Drop", "Deg Rate/Level", "Robustness Score"]
    return summary_df[cols].reset_index(drop=True)


def plot_degradation(rel_df, summary_df, model_colors, metric="accuracy"):
    """
    Two-panel figure:
      Left  — performance curves per model vs distortion level
      Right — horizontal bar chart of robustness scores with % drop annotation

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Degradation Rate Analysis — Metric: {metric.capitalize()}",
        fontsize=13, fontweight="bold"
    )

    # ── Left: performance curves ──────────────────────────────────────────────
    for model in rel_df["model_name"].unique():
        mdf   = rel_df[rel_df["model_name"] == model].sort_values("distortion_level")
        color = model_colors.get(model, "#888888")
        ax_left.plot(
            mdf["distortion_level"], mdf[metric],
            marker="o", markersize=5, linewidth=2,
            color=color, label=model
        )

    ax_left.set_xlabel("Distortion Level", fontsize=10)
    ax_left.set_ylabel(metric.capitalize(), fontsize=10)
    ax_left.set_title("Performance vs Distortion Level", fontsize=11)
    ax_left.legend(fontsize=8, loc="lower left")
    ax_left.set_ylim(0, 1.05)
    ax_left.set_xlim(left=0)
    ax_left.grid(True, alpha=0.3)

    # ── Right: robustness bar chart ───────────────────────────────────────────
    models_sorted = summary_df["Model"].tolist()
    scores        = summary_df["Robustness Score"].tolist()
    pct_drops     = summary_df["% Drop"].tolist()
    colors        = [model_colors.get(m, "#888888") for m in models_sorted]

    bars = ax_right.barh(
        models_sorted, scores,
        color=colors, edgecolor="white",
        linewidth=1.2, height=0.55
    )

    for bar, score, drop in zip(bars, scores, pct_drops):
        ax_right.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}  (drop: {drop:.1f}%)",
            va="center", fontsize=8.5, color="#333333"
        )

    ax_right.set_xlabel("Robustness Score (0–100)", fontsize=10)
    ax_right.set_title("Model Robustness Ranking", fontsize=11)
    ax_right.set_xlim(0, 120)
    ax_right.invert_yaxis()
    ax_right.grid(True, axis="x", alpha=0.3)

    # Bold the winner
    if models_sorted:
        ax_right.get_yticklabels()[0].set_fontweight("bold")

    plt.tight_layout()
    return fig


def get_robustness_tier(score):
    """Return a display tier label based on robustness score."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 55:
        return "Moderate"
    else:
        return "Fragile"
