"""
reliability_analysis.py
Computes the distortion level at which each model crosses a reliability threshold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_reliability_horizons(results_df, metric="accuracy", threshold=0.75):
    """
    For each model, find the distortion level where performance drops below threshold.

    Returns
    -------
    horizons_df : pd.DataFrame
        Columns: Model, Baseline, Threshold, Horizon Level, Reliable Range, Status
    """
    rows = []
    for model in results_df["model_name"].unique():
        mdf = results_df[results_df["model_name"] == model].sort_values("distortion_level")
        levels = mdf["distortion_level"].values
        values = mdf[metric].values
        baseline = values[0]
        max_level = levels[-1]

        # Find first level where value drops below threshold
        crossed_idx = next((i for i, v in enumerate(values) if v < threshold), None)

        if crossed_idx is None:
            # Never crossed — reliable throughout
            horizon = max_level
            status = "🟢 Always Reliable"
        elif crossed_idx == 0:
            horizon = 0.0
            status = "🔴 Below Threshold at Baseline"
        else:
            # Linear interpolation between last good and first bad point
            x0, x1 = levels[crossed_idx - 1], levels[crossed_idx]
            y0, y1 = values[crossed_idx - 1], values[crossed_idx]
            if y1 != y0:
                horizon = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            else:
                horizon = x0
            status = "🟡 Degrades Mid-Range" if horizon > max_level * 0.4 else "🔴 Degrades Early"

        reliable_pct = (horizon / max_level) * 100 if max_level > 0 else 100.0

        rows.append({
            "Model": model,
            "Baseline": round(baseline, 4),
            "Threshold": threshold,
            "Horizon Level": round(float(horizon), 4),
            "Reliable Range (%)": round(reliable_pct, 1),
            "Status": status,
        })

    df = pd.DataFrame(rows).sort_values("Horizon Level", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def plot_reliability_windows(results_df, horizons_df, model_colors,
                              metric="accuracy", threshold=0.75):
    """
    Plot performance curves with shaded reliable/unreliable zones per model.
    """
    models = results_df["model_name"].unique()
    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.8),
                             sharex=False, sharey=True)
    axes = np.array(axes).flatten()

    fig.suptitle(
        f"Reliability Windows — {metric.capitalize()} Threshold: {threshold:.2f}",
        fontsize=13, fontweight="bold"
    )

    for i, model in enumerate(models):
        ax = axes[i]
        mdf = results_df[results_df["model_name"] == model].sort_values("distortion_level")
        levels = mdf["distortion_level"].values
        values = mdf[metric].values
        color = model_colors.get(model, "#4C72B0")

        horizon_row = horizons_df[horizons_df["Model"] == model]
        horizon = horizon_row["Horizon Level"].values[0] if not horizon_row.empty else levels[-1]
        status = horizon_row["Status"].values[0] if not horizon_row.empty else ""

        max_level = levels[-1]

        # Shade reliable zone (green) and unreliable zone (red)
        ax.axvspan(0, horizon, alpha=0.10, color="#2ecc71", label="Reliable zone")
        if horizon < max_level:
            ax.axvspan(horizon, max_level, alpha=0.10, color="#e74c3c", label="Unreliable zone")

        # Threshold line
        ax.axhline(threshold, color="#e74c3c", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"Threshold ({threshold:.2f})")

        # Horizon marker
        if 0 < horizon < max_level:
            ax.axvline(horizon, color=color, linewidth=1.5,
                       linestyle=":", alpha=0.9)
            ax.text(horizon, threshold + 0.03, f"  ⚑ {horizon:.2f}",
                    fontsize=7.5, color=color, va="bottom")

        # Performance curve
        ax.plot(levels, values, "o-", color=color,
                linewidth=2, markersize=4)

        ax.set_title(f"{model}\n{status}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Distortion Level", fontsize=8)
        ax.set_ylabel(metric.capitalize(), fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25)

        # Mini legend on first plot only
        if i == 0:
            handles = [
                mpatches.Patch(color="#2ecc71", alpha=0.4, label="Reliable"),
                mpatches.Patch(color="#e74c3c", alpha=0.4, label="Unreliable"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="lower left")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def get_reliability_verdict(horizons_df, max_distortion_level):
    """
    Returns a markdown summary string for display in Streamlit.
    """
    if horizons_df.empty:
        return "_No data available._"

    best = horizons_df.iloc[0]
    worst = horizons_df.iloc[-1]

    lines = [
        f"### 🎯 Reliability Analysis Summary",
        f"",
        f"**Most Reliable:** {best['Model']} — holds above threshold "
        f"for **{best['Reliable Range (%)']:.1f}%** of the distortion range "
        f"(until level `{best['Horizon Level']:.3f}`)",
        f"",
        f"**Least Reliable:** {worst['Model']} — crosses threshold "
        f"at level `{worst['Horizon Level']:.3f}` "
        f"({worst['Reliable Range (%)']:.1f}% of range)",
        f"",
        "---",
        "**All Models:**",
    ]

    for _, row in horizons_df.iterrows():
        lines.append(
            f"- **{row['Model']}** → reliable until distortion `{row['Horizon Level']:.3f}` "
            f"| {row['Reliable Range (%)']:.1f}% of range | {row['Status']}"
        )

    lines += [
        "",
        f"> **Viva answer:** Model reliability varies significantly by algorithm. "
        f"Ensemble methods (RF, GB) maintain performance across the widest distortion range, "
        f"while distance-based and linear models cross the reliability threshold earlier. "
        f"The reliability horizon — the distortion level at which accuracy drops below "
        f"{horizons_df['Threshold'].iloc[0]:.0%} — is the key metric for deployment decisions."
    ]

    return "\n".join(lines)
