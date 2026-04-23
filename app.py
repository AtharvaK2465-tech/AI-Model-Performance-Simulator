import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from model import load_dataset, train_models
from distortions import apply_distortion
from evaluation import evaluate

st.set_page_config(page_title="AI Model Performance Simulator", layout="wide")
st.title("🤖 AI Model Performance Simulator")
st.markdown("Simulate how ML models degrade under real-world data distortions.")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

# Dataset selection
dataset_source = st.sidebar.radio("Dataset Source", ["Built-in Dataset", "Upload CSV"])

X, y, ds_name = None, None, None

if dataset_source == "Built-in Dataset":
    dataset_name = st.sidebar.selectbox("Dataset", ["iris", "wine", "breast_cancer"])
    try:
        X, y, ds_name = load_dataset(name=dataset_name)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10), width='stretch')

            columns = df.columns.tolist()
            target_col = st.sidebar.selectbox("Select Target Column", columns, index=len(columns) - 1)
            feature_cols = [c for c in columns if c != target_col]

            st.sidebar.markdown(f"**Features:** {len(feature_cols)} columns")
            st.sidebar.markdown(f"**Target:** `{target_col}`")

            X = df[feature_cols].values.astype(float)
            y_raw = df[target_col].values
            if y_raw.dtype == object:
                classes = np.unique(y_raw)
                mapping = {c: i for i, c in enumerate(classes)}
                y = np.array([mapping[v] for v in y_raw])
            else:
                y = y_raw.astype(int)
            ds_name = uploaded_file.name

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.info("Make sure your CSV has numeric features and a valid target column.")

# Distortion controls
st.sidebar.header("🎛️ Distortion Settings")
levels = st.sidebar.slider("Max Distortion Level", 0.1, 1.0, 0.4, step=0.1)
num_levels = st.sidebar.slider("Number of Levels", 3, 10, 5)

# Run button
if st.sidebar.button("▶ Run Simulation"):
    if X is None or y is None:
        st.warning("⚠️ Please select or upload a dataset first.")
    else:
        distortion_levels = list(np.linspace(0, levels, num_levels))

        with st.spinner("Training models on clean data..."):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                rf, lr, svm = train_models(X_train, y_train)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

        st.success(f"✅ Models trained on **{ds_name}** | Samples: {len(y)} | Features: {X.shape[1]}")

        rf_results, lr_results, svm_results = [], [], []

        progress = st.progress(0)
        status = st.empty()

        for i, level in enumerate(distortion_levels):
            status.text(f"Evaluating distortion level {round(level, 2)}...")
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

        status.text("✅ Simulation complete!")

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

        # Results table
        st.subheader("📊 Results Table")
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
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, width='stretch')

        # Download button
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="simulation_results.csv",
            mime="text/csv"
        )
