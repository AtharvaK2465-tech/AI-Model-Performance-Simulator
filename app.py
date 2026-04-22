import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import load_dataset, train_models
from distortions import apply_distortion
from evaluation import evaluate

st.set_page_config(page_title="AI Model Performance Simulator", layout="wide")
st.title("🤖 AI Model Performance Simulator")
st.markdown("Simulate how ML models degrade under real-world data distortions.")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
dataset_name = st.sidebar.selectbox("Dataset", ["iris", "wine", "breast_cancer"])
levels = st.sidebar.slider("Max Distortion Level", 0.1, 1.0, 0.4, step=0.1)
num_levels = st.sidebar.slider("Number of Levels", 3, 10, 5)

if st.sidebar.button("▶ Run Simulation"):
    distortion_levels = list(np.linspace(0, levels, num_levels))

    with st.spinner("Loading dataset and training models..."):
        X, y, _ = load_dataset(name=dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf, lr, svm = train_models(X_train, y_train)

    st.success(f"✅ Models trained on **{dataset_name}** | Samples: {len(y)} | Features: {X.shape[1]}")

    rf_results, lr_results, svm_results = [], [], []

    progress = st.progress(0)
    for i, level in enumerate(distortion_levels):
        X_dist, y_dist = apply_distortion(
            X_test.copy(), y_test.copy(),
            noise_level=level,
            drift_level=level,
            dist_shift_level=level,
            imbalance_level=level
        )
        rf_results.append(evaluate(rf,  X_dist, y_dist))
        lr_results.append(evaluate(lr,  X_dist, y_dist))
        svm_results.append(evaluate(svm, X_dist, y_dist))
        progress.progress((i + 1) / len(distortion_levels))

    # Plot
    METRICS = ["accuracy", "precision", "recall", "f1"]
    TITLES  = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Model Performance Under Increasing Distortion", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        ax.plot(distortion_levels, [r[metric] for r in rf_results],  marker='o', label='Random Forest',       color='steelblue')
        ax.plot(distortion_levels, [r[metric] for r in lr_results],  marker='s', label='Logistic Regression', color='tomato')
        ax.plot(distortion_levels, [r[metric] for r in svm_results], marker='^', label='SVM',                 color='seagreen')
        ax.set_title(title)
        ax.set_xlabel("Distortion Level")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

    # Table
    st.subheader("📊 Results Table")
    import pandas as pd
    rows = []
    for i, level in enumerate(distortion_levels):
        rows.append({
            "Level":   round(level, 3),
            "RF Acc":  round(rf_results[i]["accuracy"],  3),
            "RF F1":   round(rf_results[i]["f1"],        3),
            "LR Acc":  round(lr_results[i]["accuracy"],  3),
            "LR F1":   round(lr_results[i]["f1"],        3),
            "SVM Acc": round(svm_results[i]["accuracy"], 3),
            "SVM F1":  round(svm_results[i]["f1"],       3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
