import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import load_dataset, train_models
from distortions import apply_distortion
from evaluation import evaluate

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Model Performance Simulator", layout="wide")
st.title("🤖 AI Model Performance Simulator")
st.markdown("Simulate how ML models degrade under real-world data distortions.")

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
dataset_source = st.sidebar.radio("Dataset Source", ["Built-in Dataset", "Upload CSV"])

X, y, ds_name = None, None, None

# ─── Built-in Dataset ──────────────────────────────────────────────────────────
if dataset_source == "Built-in Dataset":
    dataset_name = st.sidebar.selectbox("Dataset", ["iris", "wine", "breast_cancer"])
    try:
        X, y, ds_name = load_dataset(name=dataset_name)
        st.sidebar.success(f"✅ Loaded: {dataset_name}")
        st.sidebar.markdown(f"**Samples:** {len(y)}")
        st.sidebar.markdown(f"**Features:** {X.shape[1]}")
        st.sidebar.markdown(f"**Classes:** {len(np.unique(y))}")
    except Exception as e:
        st.sidebar.error(f"Failed to load dataset: {e}")

# ─── CSV Upload ────────────────────────────────────────────────────────────────
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Drop completely empty rows/columns
            df.dropna(axis=1, how='all', inplace=True)
            df.dropna(axis=0, how='all', inplace=True)

            if df.empty:
                st.error("❌ Uploaded CSV is empty after cleaning.")
                st.stop()

            # Show preview
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # Target column selection
            columns = df.columns.tolist()
            default_target_idx = len(columns) - 1
            target_col = st.sidebar.selectbox(
                "Select Target Column",
                columns,
                index=default_target_idx
            )

            feature_cols = [c for c in columns if c != target_col]
            feature_df = df[feature_cols].copy()

            # ── Feature Processing ──
            non_numeric_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

            # Auto-encode categorical columns with low cardinality
            encoded_cols = []
            if non_numeric_cols:
                for col in non_numeric_cols:
                    unique_vals = feature_df[col].dropna().unique()
                    if len(unique_vals) <= 20:
                        col_le = LabelEncoder()
                        feature_df[col] = feature_df[col].astype(str)
                        feature_df[col] = col_le.fit_transform(feature_df[col])
                        encoded_cols.append(col)

                dropped_cols = [c for c in non_numeric_cols if c not in encoded_cols]
                if encoded_cols:
                    st.sidebar.info(f"ℹ️ Auto-encoded categorical columns: {encoded_cols}")
                if dropped_cols:
                    st.sidebar.warning(f"⚠️ Dropped high-cardinality columns: {dropped_cols}")

            all_feature_cols = numeric_cols + encoded_cols

            if len(all_feature_cols) == 0:
                st.error(
                    "❌ No usable feature columns found after processing. "
                    "Please upload a dataset with numeric or low-cardinality categorical features."
                )
                st.stop()

            # Drop rows with NaN
            feature_df = feature_df[all_feature_cols].copy()
            before = len(feature_df)
            feature_df.dropna(inplace=True)
            dropped_rows = before - len(feature_df)
            if dropped_rows > 0:
                st.sidebar.info(f"ℹ️ Dropped {dropped_rows} rows with missing values.")

            X = feature_df.values.astype(float)

            st.sidebar.markdown(
                f"**Features used:** {len(all_feature_cols)} columns "
                f"({len(numeric_cols)} numeric, {len(encoded_cols)} encoded)"
            )

            # ── Target Processing ──
            y_raw = df.loc[feature_df.index, target_col].values

            # Handle NaN in target
            target_nan_mask = pd.isnull(y_raw)
            if target_nan_mask.any():
                count = target_nan_mask.sum()
                st.sidebar.info(f"ℹ️ Dropped {count} rows with missing target values.")
                X = X[~target_nan_mask]
                y_raw = y_raw[~target_nan_mask]

            # Encode target
            le = LabelEncoder()
            try:
                y = le.fit_transform(y_raw.astype(str))
            except Exception as enc_err:
                st.error(f"❌ Could not encode target column: {enc_err}")
                st.stop()

            n_classes = len(np.unique(y))
            n_samples = len(y)

            # Validate minimum samples
            if n_samples < 20:
                st.error(
                    f"❌ Too few samples ({n_samples}). "
                    "Need at least 20 samples to run simulation."
                )
                st.stop()

            # Validate minimum classes
            if n_classes < 2:
                st.error(
                    "❌ Target column has only 1 class. "
                    "Need at least 2 classes for classification."
                )
                st.stop()

            # Validate samples per class for stratified split
            min_class_count = min(np.sum(y == c) for c in np.unique(y))
            if min_class_count < 2:
                st.error(
                    "❌ At least one class has fewer than 2 samples. "
                    "Please use a dataset with more samples per class."
                )
                st.stop()

            ds_name = uploaded_file.name

            st.sidebar.success(f"✅ CSV loaded: {ds_name}")
            st.sidebar.markdown(f"**Samples:** {n_samples}")
            st.sidebar.markdown(f"**Classes:** {n_classes} → {list(le.classes_[:5])}"
                                + ("..." if n_classes > 5 else ""))

        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty.")
            st.stop()
        except pd.errors.ParserError:
            st.error("❌ Could not parse CSV. Make sure it is a valid comma-separated file.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error reading CSV: {e}")
            st.stop()

    else:
        st.info("👈 Upload a CSV file from the sidebar to get started, or switch to a built-in dataset.")

# ─── Distortion Settings ───────────────────────────────────────────────────────
st.sidebar.header("🎛️ Distortion Settings")
max_level = st.sidebar.slider("Max Distortion Level", 0.1, 1.0, 0.4, step=0.1)
num_levels = st.sidebar.slider("Number of Levels", 3, 10, 5)

# ─── Run Simulation ────────────────────────────────────────────────────────────
if st.sidebar.button("▶ Run Simulation"):

    if X is None or y is None:
        st.warning("⚠️ Please select or upload a valid dataset first.")
        st.stop()

    distortion_levels = list(np.linspace(0, max_level, num_levels))

    # Train models
    with st.spinner("Training models on clean data..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            rf, lr, svm = train_models(X_train, y_train)
        except ValueError as ve:
            st.error(f"❌ Train/test split failed: {ve}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            st.stop()

    st.success(
        f"✅ Models trained on **{ds_name}** | "
        f"Samples: {len(y)} | Features: {X.shape[1]} | "
        f"Classes: {len(np.unique(y))}"
    )

    rf_results, lr_results, svm_results = [], [], []
    progress = st.progress(0)
    status = st.empty()

    for i, level in enumerate(distortion_levels):
        status.text(f"Evaluating distortion level {round(level, 3)}...")
        try:
            X_dist, y_dist = apply_distortion(
                X_test.copy(), y_test.copy(),
                noise_level=level,
                drift_level=level,
                dist_shift_level=level,
                imbalance_level=level
            )

            if len(np.unique(y_dist)) < 2:
                st.warning(
                    f"⚠️ Level {round(level, 2)}: Only 1 class left after imbalance — skipping."
                )
                rf_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
                lr_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
                svm_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
            else:
                rf_results.append(evaluate(rf,  X_dist, y_dist))
                lr_results.append(evaluate(lr,  X_dist, y_dist))
                svm_results.append(evaluate(svm, X_dist, y_dist))

        except Exception as e:
            st.warning(f"⚠️ Error at level {round(level, 2)}: {e} — skipping.")
            rf_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
            lr_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})
            svm_results.append({"accuracy": 0, "precision": 0, "recall": 0, "f1": 0})

        progress.progress((i + 1) / len(distortion_levels))

    status.text("✅ Simulation complete!")

    # ─── Plot ──────────────────────────────────────────────────────────────────
    METRICS = ["accuracy", "precision", "recall", "f1"]
    TITLES  = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 Score (macro)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Model Performance Under Increasing Distortion\nDataset: {ds_name}",
        fontsize=13, fontweight="bold"
    )

    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        ax.plot(distortion_levels, [r[metric] for r in rf_results],
                marker='o', label='Random Forest',       color='steelblue')
        ax.plot(distortion_levels, [r[metric] for r in lr_results],
                marker='s', label='Logistic Regression', color='tomato')
        ax.plot(distortion_levels, [r[metric] for r in svm_results],
                marker='^', label='SVM',                 color='seagreen')
        ax.set_title(title)
        ax.set_xlabel("Distortion Level")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

    # ─── Results Table ─────────────────────────────────────────────────────────
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
    st.dataframe(results_df, use_container_width=True)

    # ─── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv"
    )
