import matplotlib.pyplot as plt
import os


METRICS = ["accuracy", "precision", "recall", "f1"]
TITLES  = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)"]


def plot_results(levels, rf_results, lr_results, svm_results, save_path="results.png"):
    """
    rf_results / lr_results / svm_results: list of dicts, one per distortion level.
    Each dict has keys: accuracy, precision, recall, f1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Model Performance Under Increasing Distortion", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        rf_vals  = [r[metric] for r in rf_results]
        lr_vals  = [r[metric] for r in lr_results]
        svm_vals = [r[metric] for r in svm_results]
        ax.plot(levels, rf_vals,  marker='o', label='Random Forest',        color='steelblue')
        ax.plot(levels, lr_vals,  marker='s', label='Logistic Regression',  color='tomato')
        ax.plot(levels, svm_vals, marker='^', label='SVM',                  color='seagreen')
        ax.set_title(title)
        ax.set_xlabel("Distortion Level")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[✓] Plot saved to: {os.path.abspath(save_path)}")
    plt.show()
