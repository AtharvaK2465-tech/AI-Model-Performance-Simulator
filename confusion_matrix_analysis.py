"""
confusion_matrix_analysis.py
Generates confusion matrices for each model at clean baseline vs max distortion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


def compute_confusion_matrices(trained_models, X_train, y_train,
                                X_test_clean, y_test,
                                X_test_distorted, y_test_distorted=None):
    """
    For each model, compute confusion matrices at clean and max-distortion level.

    Parameters
    ----------
    trained_models   : dict   {model_name: fitted_estimator}
    X_train          : array  (unused - models already trained, kept for signature)
    y_train          : array  (unused - kept for signature)
    X_test_clean     : array  Clean test features
    y_test           : array  True labels for clean test set
    X_test_distorted : array  Max-distortion test features
    y_test_distorted : array  True labels for distorted test set (may differ from
                              y_test when class imbalance distortion drops samples)

    Returns
    -------
    results : dict  {model_name: {"clean": cm, "distorted": cm, "classes": array}}
    """
    if y_test_distorted is None:
        y_test_distorted = y_test

    results     = {}
    all_classes = np.unique(np.concatenate([y_test, y_test_distorted]))

    for name, model in trained_models.items():
        try:
            pred_clean = model.predict(X_test_clean)
            pred_dist  = model.predict(X_test_distorted)

            cm_clean = confusion_matrix(y_test,           pred_clean, labels=all_classes)
            cm_dist  = confusion_matrix(y_test_distorted, pred_dist,  labels=all_classes)

            results[name] = {
                "clean":     cm_clean,
                "distorted": cm_dist,
                "classes":   all_classes,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def plot_confusion_matrices(cm_results, model_colors):
    """
    Plot a row of confusion matrix pairs (clean | distorted) per model.

    Returns
    -------
    fig : matplotlib Figure
    """
    valid_models = [k for k, v in cm_results.items() if "error" not in v]
    n = len(valid_models)

    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No confusion matrices to display.",
                ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        n, 2,
        figsize=(10, n * 3.8),
        squeeze=False
    )

    fig.suptitle(
        "Confusion Matrices — Clean Baseline vs Max Distortion",
        fontsize=13, fontweight="bold", y=1.01
    )

    for row, model_name in enumerate(valid_models):
        data    = cm_results[model_name]
        classes = data["classes"]
        color   = model_colors.get(model_name, "#4C72B0")

        for col, (cm, tag) in enumerate([
            (data["clean"],     "Clean Baseline"),
            (data["distorted"], "Max Distortion"),
        ]):
            ax    = axes[row][col]
            shade = color if col == 0 else _darken(color, 0.60)
            _draw_cm(ax, cm, classes, shade)
            ax.set_title(
                f"{model_name}  |  {tag}",
                fontsize=9, fontweight="bold", pad=6
            )

    plt.tight_layout()
    return fig


def _draw_cm(ax, cm, classes, color):
    """Render a single confusion matrix heatmap."""
    n        = len(classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm  = cm.astype(float) / (row_sums + 1e-9)

    # Convert color to hex so LinearSegmentedColormap always gets a valid input
    try:
        hex_color = mcolors.to_hex(color)
    except Exception:
        hex_color = "#4C72B0"

    cmap = LinearSegmentedColormap.from_list("model_cm", ["#ffffff", hex_color])
    ax.imshow(norm_cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    tick_labels = [str(c) for c in classes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("Actual", fontsize=8)

    thresh = norm_cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            pct        = norm_cm[i, j]
            count      = cm[i, j]
            cell_color = "white" if pct > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})",
                    ha="center", va="center",
                    fontsize=7, color=cell_color)

    acc = np.trace(cm) / (cm.sum() + 1e-9)
    ax.set_xlabel(f"Predicted   |   Acc: {acc:.1%}", fontsize=8)


def _darken(color, factor=0.65):
    """
    Darken any matplotlib-compatible color (named, hex, rgb tuple) by factor.
    Uses matplotlib.colors.to_hex to handle all color formats safely.
    """
    try:
        hex_color = mcolors.to_hex(color).lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return color  # fallback - return original if anything fails
