"""
recommendation_engine.py

Composite model scoring and recommendation dashboard.

Scoring formula (weights normalised to sum=1):
    composite = w_robustness * robustness_norm
              + w_peak       * peak_accuracy_norm
              + w_stability  * stability_norm

  - robustness_norm  : AUC under accuracy curve, scaled to [0,1]
  - peak_accuracy    : best accuracy across all distortion levels
  - stability_norm   : 1 - (std of accuracy across levels), scaled to [0,1]

Returns
-------
fig_rec : matplotlib Figure  — radar chart + composite bar chart
rec_df  : pd.DataFrame       — one row per model, all sub-scores + composite
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from math import pi


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_scores(rel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw sub-scores for every model in rel_df."""
    rows = []
    for model_name, grp in rel_df.groupby("model_name"):
        grp = grp.sort_values("distortion_level")
        levels = grp["distortion_level"].values
        accs   = grp["accuracy"].values

        # Robustness: normalised AUC under accuracy curve
        if len(levels) > 1:
            span        = levels[-1] - levels[0]
            auc_raw     = np.trapezoid(accs, levels) if hasattr(np, "trapezoid") \
                          else np.trapz(accs, levels)
            robustness  = (auc_raw / span) * 100 if span > 0 else accs.mean() * 100
        else:
            robustness = accs[0] * 100

        peak_accuracy = float(accs.max())
        stability     = float(1.0 - accs.std())   # higher = more stable

        rows.append({
            "Model":           model_name,
            "Robustness Score": round(robustness, 2),
            "Peak Accuracy":    round(peak_accuracy, 4),
            "Stability Score":  round(stability, 4),
        })

    return pd.DataFrame(rows)


def _normalise_col(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def _composite(scores_df, w_robustness, w_peak, w_stability):
    total = w_robustness + w_peak + w_stability
    if total < 1e-9:
        total = 1.0
    wr, wp, ws = w_robustness / total, w_peak / total, w_stability / total

    df = scores_df.copy()
    df["_rob_n"]  = _normalise_col(df["Robustness Score"])
    df["_peak_n"] = _normalise_col(df["Peak Accuracy"])
    df["_stab_n"] = _normalise_col(df["Stability Score"])
    df["Composite Score"] = (
        wr * df["_rob_n"] + wp * df["_peak_n"] + ws * df["_stab_n"]
    ).round(4)
    df = df.drop(columns=["_rob_n", "_peak_n", "_stab_n"])
    return df.sort_values("Composite Score", ascending=False).reset_index(drop=True)


# ── Radar chart ───────────────────────────────────────────────────────────────

def _radar_chart(ax, rec_df, model_colors):
    categories  = ["Robustness", "Peak Accuracy", "Stability", "Composite"]
    col_map     = {
        "Robustness":    "Robustness Score",
        "Peak Accuracy": "Peak Accuracy",
        "Stability":     "Stability Score",
        "Composite":     "Composite Score",
    }
    N    = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    # normalise each column to [0,1] for radar display
    disp = rec_df.copy()
    for cat, col in col_map.items():
        disp[col] = _normalise_col(disp[col])

    for _, row in disp.iterrows():
        values  = [row[col_map[c]] for c in categories]
        values += values[:1]
        color   = model_colors.get(row["Model"], "#888888")
        ax.plot(angles, values, linewidth=2, linestyle="solid",
                label=row["Model"], color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.set_title("Model Comparison Radar", fontsize=11,
                 fontweight="bold", pad=18)


# ── Bar chart ─────────────────────────────────────────────────────────────────

def _bar_chart(ax, rec_df, model_colors):
    models  = rec_df["Model"].tolist()
    scores  = rec_df["Composite Score"].tolist()
    colors  = [model_colors.get(m, "#888888") for m in models]
    x       = np.arange(len(models))

    bars = ax.bar(x, scores, color=colors, width=0.55, edgecolor="white",
                  linewidth=1.2, zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Composite Score", fontsize=10)
    ax.set_title("Composite Score Ranking", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Highlight top model
    if bars:
        bars[0].set_edgecolor("gold")
        bars[0].set_linewidth(2.5)


# ── Top-3 trajectory mini-plots ───────────────────────────────────────────────

def _trajectory_plots(axes_row, rec_df, rel_df, model_colors, n=3):
    top_models = rec_df["Model"].tolist()[:n]
    for ax, model_name in zip(axes_row, top_models):
        grp    = rel_df[rel_df["model_name"] == model_name].sort_values("distortion_level")
        levels = grp["distortion_level"].values
        color  = model_colors.get(model_name, "#888888")

        for metric, ls, alpha in [
            ("accuracy",  "-",  1.0),
            ("f1",        "--", 0.75),
            ("roc_auc",   ":",  0.75),
        ]:
            if metric in grp.columns:
                ax.plot(levels, grp[metric].values,
                        linestyle=ls, color=color, alpha=alpha,
                        linewidth=1.6, label=metric.upper())

        ax.set_title(model_name, fontsize=9, fontweight="bold", color=color)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distortion Level", fontsize=7)
        ax.set_ylabel("Score", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes if fewer than n models
    for ax in axes_row[len(top_models):]:
        ax.set_visible(False)


# ── Public API ────────────────────────────────────────────────────────────────

def build_recommendation(
    rel_df: pd.DataFrame,
    model_colors: dict,
    w_robustness: float = 0.40,
    w_peak:       float = 0.35,
    w_stability:  float = 0.25,
):
    """
    Build the full recommendation dashboard figure and scored DataFrame.

    Parameters
    ----------
    rel_df        : long-form DataFrame from app.py (model_name, distortion_level, metrics…)
    model_colors  : dict mapping model name → hex colour
    w_robustness  : weight for robustness sub-score  (will be normalised)
    w_peak        : weight for peak accuracy sub-score
    w_stability   : weight for stability sub-score

    Returns
    -------
    fig : matplotlib Figure
    rec_df : pd.DataFrame  sorted by Composite Score descending
    """
    scores_df = _compute_scores(rel_df)
    rec_df    = _composite(scores_df, w_robustness, w_peak, w_stability)

    n_models  = len(rec_df)
    n_traj    = min(n_models, 3)

    # ── Layout: radar | bar  /  trajectory row ────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Model Recommendation Engine — Composite Scoring Dashboard",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs_top = gridspec.GridSpec(1, 2, figure=fig,
                               left=0.05, right=0.97,
                               top=0.90, bottom=0.48,
                               wspace=0.35)
    gs_bot = gridspec.GridSpec(1, n_traj, figure=fig,
                               left=0.05, right=0.97,
                               top=0.40, bottom=0.07,
                               wspace=0.30)

    ax_radar = fig.add_subplot(gs_top[0, 0], polar=True)
    ax_bar   = fig.add_subplot(gs_top[0, 1])
    traj_axes = [fig.add_subplot(gs_bot[0, i]) for i in range(n_traj)]

    _radar_chart(ax_radar, rec_df, model_colors)
    _bar_chart(ax_bar, rec_df, model_colors)
    _trajectory_plots(traj_axes, rec_df, rel_df, model_colors, n=n_traj)

    # Section label
    fig.text(0.05, 0.43, "Top-3 Model Trajectories (Accuracy / F1 / ROC-AUC vs Distortion)",
             fontsize=10, fontweight="bold", color="#333333")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, rec_df
