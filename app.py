import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

from model import load_dataset, train_models, ALL_MODELS, MODEL_COLORS, MODEL_MARKERS
from distortions import apply_distortion
from evaluation import evaluate
from run_logger import save_run, load_all_runs

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Model Performance Simulator", layout="wide")
st.title("🤖 AI Model Performance Simulator")
st.markdown("Simulate how ML models degrade under real-world data distortions.")

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
dataset_source = st.sidebar.radio("Dataset Source", ["Built-in Dataset", "Upload CSV"])

X, y, ds_name = None, None, None

# ─── Helper: Detect Regression Target ──────────────────────────────────────────
def is_regression_target(values, threshold=0.05):
    try:
        vals = pd.to_numeric(values, errors='coerce')
        if vals.isnull().all():
            return False
        unique_ratio = len(vals.dropna().unique()) / len(vals.dropna())
        return unique_ratio > threshold and len(vals.dropna().unique()) > 20
    except Exception:
        return False

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
            df = pd.read_csv(uploaded_file)
            df.dropna(axis=1, how='all', inplace=True)
            df.dropna(axis=0, how='all', inplace=True)

            if df.empty:
                st.error("❌ Uploaded CSV is empty after cleaning.")
                st.stop()
            if len(df) < 20:
                st.error(f"❌ Dataset has only {len(df)} rows. Need at least 20 samples.")
                st.stop()
            if len(df.columns) < 2:
                st.error("❌ Dataset needs at least 2 columns (1 feature + 1 target).")
                st.stop()

            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            columns    = df.columns.tolist()
            target_col = st.sidebar.selectbox(
                "Select Target Column", columns, index=len(columns) - 1
            )

            if is_regression_target(df[target_col].values):
                st.warning(
                    f"⚠️ The column **'{target_col}'** looks like a continuous/regression target "
                    f"({df[target_col].nunique()} unique values). "
                    "This simulator is for **classification only**. "
                    "Please select a categorical target column."
                )

            feature_cols     = [c for c in columns if c != target_col]
            feature_df       = df[feature_cols].copy()
            non_numeric_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
            numeric_cols     = feature_df.select_dtypes(include=[np.number]).columns.tolist()

            id_like_cols = [
                col for col in numeric_cols
                if feature_df[col].nunique() == len(feature_df)
            ]
            if id_like_cols:
                numeric_cols = [c for c in numeric_cols if c not in id_like_cols]
                st.sidebar.warning(f"⚠️ Dropped ID-like columns: {id_like_cols}")

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
                    st.sidebar.info(f"ℹ️ Auto-encoded: {encoded_cols}")
                if dropped_cols:
                    st.sidebar.warning(f"⚠️ Dropped high-cardinality columns: {dropped_cols}")

            all_feature_cols = numeric_cols + encoded_cols

            if len(all_feature_cols) == 0:
                st.error("❌ No usable feature columns found.")
                st.stop()
            if len(all_feature_cols) == 1:
                st.warning("⚠️ Only 1 feature column found. Results may be unreliable.")

            feature_df   = feature_df[all_feature_cols].copy()
            before_dedup = len(feature_df)
            feature_df   = feature_df.drop_duplicates()
            dupes        = before_dedup - len(feature_df)
            if dupes > 0:
                st.sidebar.info(f"ℹ️ Removed {dupes} duplicate rows.")

            before_nan  = len(feature_df)
            feature_df.dropna(inplace=True)
            dropped_nan = before_nan - len(feature_df)
            if dropped_nan > 0:
                st.sidebar.info(f"ℹ️ Dropped {dropped_nan} rows with missing values.")

            if len(feature_df) < 20:
                st.error(f"❌ Only {len(feature_df)} usable rows remain. Need at least 20.")
                st.stop()

            X = feature_df.values.astype(float)
            st.sidebar.markdown(
                f"**Features used:** {len(all_feature_cols)} "
                f"({len(numeric_cols)} numeric, {len(encoded_cols)} encoded)"
            )

            y_raw           = df.loc[feature_df.index, target_col].values
            target_nan_mask = pd.isnull(y_raw)
            if target_nan_mask.any():
                count = target_nan_mask.sum()
                st.sidebar.info(f"ℹ️ Dropped {count} rows with missing target values.")
                X     = X[~target_nan_mask]
                y_raw = y_raw[~target_nan_mask]

            le = LabelEncoder()
            try:
                y = le.fit_transform(y_raw.astype(str))
            except Exception as enc_err:
                st.error(f"❌ Could not encode target column: {enc_err}")
                st.stop()

            n_classes = len(np.unique(y))
            n_samples = len(y)

            if n_samples < 20:
                st.error(f"❌ Too few samples ({n_samples}). Need at least 20.")
                st.stop()
            if n_classes < 2:
                st.error(f"❌ Target column has only 1 class. Need at least 2.")
                st.stop()
            if n_classes > 50:
                st.error(f"❌ Target column has {n_classes} classes — looks like regression/ID.")
                st.stop()

            class_counts    = Counter(y)
            min_class_count = min(class_counts.values())
            max_class_count = max(class_counts.values())
            imbalance_ratio = max_class_count / min_class_count

            if min_class_count < 2:
                st.error(f"❌ At least one class has only {min_class_count} sample(s).")
                st.stop()
            if imbalance_ratio > 10:
                st.warning(f"⚠️ Dataset is highly imbalanced (ratio {imbalance_ratio:.1f}x).")

            ds_name = uploaded_file.name
            st.sidebar.success(f"✅ CSV loaded: {ds_name}")
            st.sidebar.markdown(f"**Samples:** {n_samples}")
            st.sidebar.markdown(
                f"**Classes:** {n_classes} → {list(le.classes_[:5])}"
                + ("..." if n_classes > 5 else "")
            )

        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty.")
            st.stop()
        except pd.errors.ParserError:
            st.error("❌ Could not parse CSV.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error reading CSV: {e}")
            st.stop()
    else:
        st.info("👈 Upload a CSV file from the sidebar, or switch to a built-in dataset.")

# ─── Model Selection ───────────────────────────────────────────────────────────
st.sidebar.header("🤖 Model Selection")
st.sidebar.markdown("Select models to compare:")

model_selections = {}
for model_name in ALL_MODELS.keys():
    default = model_name in ["Random Forest", "Logistic Regression", "SVM"]
    model_selections[model_name] = st.sidebar.checkbox(
        model_name, value=default
    )

selected_models = [k for k, v in model_selections.items() if v]

if len(selected_models) == 0:
    st.sidebar.warning("⚠️ No models selected. Please select at least one.")
elif len(selected_models) == 1:
    st.sidebar.info("ℹ️ Select at least 2 models for a meaningful comparison.")

# ─── Simulation Settings ───────────────────────────────────────────────────────
st.sidebar.header("🎛️ Simulation Settings")

max_level = st.sidebar.slider(
    "Max Distortion Level", 0.1, 1.0, 0.4, step=0.1,
    help="How severe the distortions get. 0.1 = mild, 1.0 = extreme."
)
num_levels = st.sidebar.slider(
    "Number of Distortion Levels", 3, 10, 5,
    help="How many steps between 0 and max distortion level."
)
test_size = st.sidebar.slider(
    "Test Set Size", 0.1, 0.4, 0.2, step=0.05,
    help="Fraction of data held out for testing. Default 0.2 = 20%."
)
scale_data = st.sidebar.checkbox(
    "Scale Features (StandardScaler)", value=True,
    help="Normalize features before training. Recommended for LR and SVM."
)
random_seed = st.sidebar.number_input(
    "Random Seed", min_value=0, max_value=9999, value=42, step=1,
    help="Set seed for reproducibility."
)

# ─── Distortion Type Selection ─────────────────────────────────────────────────
st.sidebar.header("🧪 Distortion Types")
st.sidebar.markdown("Select which distortions to apply:")

use_noise       = st.sidebar.checkbox("Gaussian Noise",     value=True,
    help="Adds random Gaussian noise to all feature values.")
use_drift       = st.sidebar.checkbox("Covariate Drift",    value=True,
    help="Shifts the mean of all features.")
use_dist_shift  = st.sidebar.checkbox("Distribution Shift", value=True,
    help="Expands/compresses feature variance.")
use_imbalance   = st.sidebar.checkbox("Class Imbalance",    value=True,
    help="Progressively drops minority class samples.")
use_missing     = st.sidebar.checkbox("Missing Values",     value=False,
    help="Randomly injects NaN into features.")
use_label_noise = st.sidebar.checkbox("Label Noise",        value=False,
    help="Randomly flips class labels.")
use_outliers    = st.sidebar.checkbox("Outlier Injection",  value=False,
    help="Injects extreme values into random samples.")
use_corruption  = st.sidebar.checkbox("Feature Corruption", value=False,
    help="Zeros out random feature columns.")

if not any([use_noise, use_drift, use_dist_shift, use_imbalance,
            use_missing, use_label_noise, use_outliers, use_corruption]):
    st.sidebar.warning("⚠️ No distortions selected. Results will be flat lines.")

# ─── Run Simulation ────────────────────────────────────────────────────────────
if st.sidebar.button("▶ Run Simulation"):

    if X is None or y is None:
        st.warning("⚠️ Please select or upload a valid dataset first.")
        st.stop()

    if len(selected_models) == 0:
        st.warning("⚠️ Please select at least one model.")
        st.stop()

    distortion_levels = list(np.linspace(0, max_level, num_levels))

    X_scaled = X.copy()
    if scale_data:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_scaled)

    with st.spinner("Training models on clean data..."):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=test_size,
                random_state=int(random_seed),
                stratify=y
            )
            trained_models = train_models(X_train, y_train, selected_models)
        except ValueError as ve:
            st.error(f"❌ Train/test split failed: {ve}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Model training failed: {e}")
            st.stop()

    st.success(
        f"✅ Models trained on **{ds_name}** | "
        f"Samples: {len(y)} | Features: {X.shape[1]} | "
        f"Classes: {len(np.unique(y))} | "
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Models: {', '.join(selected_models)}"
    )

    empty_result = {
        "accuracy": 0, "precision": 0, "recall": 0,
        "f1": 0, "roc_auc": 0, "confidence": 0
    }

    # results dict: {model_name: [result_per_level]}
    all_results = {name: [] for name in selected_models}
    progress = st.progress(0)
    status   = st.empty()

    for i, level in enumerate(distortion_levels):
        status.text(f"Evaluating distortion level {round(level, 3)}...")
        try:
            X_dist, y_dist = apply_distortion(
                X_test.copy(), y_test.copy(),
                noise_level       = level if use_noise       else 0.0,
                drift_level       = level if use_drift       else 0.0,
                dist_shift_level  = level if use_dist_shift  else 0.0,
                imbalance_level   = level if use_imbalance   else 0.0,
                missing_level     = level if use_missing     else 0.0,
                label_noise_level = level if use_label_noise else 0.0,
                outlier_level     = level if use_outliers    else 0.0,
                corruption_level  = level if use_corruption  else 0.0,
            )

            if len(X_dist) == 0 or len(np.unique(y_dist)) < 2:
                st.warning(f"⚠️ Level {round(level, 2)}: Not enough class diversity — skipping.")
                for name in selected_models:
                    all_results[name].append(empty_result.copy())
            else:
                for name, model in trained_models.items():
                    all_results[name].append(evaluate(model, X_dist, y_dist))

        except Exception as e:
            st.warning(f"⚠️ Error at level {round(level, 2)}: {e} — skipping.")
            for name in selected_models:
                all_results[name].append(empty_result.copy())

        progress.progress((i + 1) / len(distortion_levels))

    status.text("✅ Simulation complete!")

    # ─── Plot ──────────────────────────────────────────────────────────────────
    METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "confidence"]
    TITLES  = [
        "Accuracy", "Precision (macro)", "Recall (macro)",
        "F1 Score (macro)", "ROC-AUC Score", "Model Confidence",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Model Performance Under Increasing Distortion\nDataset: {ds_name}",
        fontsize=13, fontweight="bold"
    )

    for ax, metric, title in zip(axes.flat, METRICS, TITLES):
        for name in selected_models:
            ax.plot(
                distortion_levels,
                [r[metric] for r in all_results[name]],
                marker=MODEL_MARKERS.get(name, "o"),
                label=name,
                color=MODEL_COLORS.get(name, "gray")
            )
        ax.set_title(title)
        ax.set_xlabel("Distortion Level")
        ax.set_ylabel(metric.replace("_", " ").capitalize())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

    # ─── Dataset & Run Info ────────────────────────────────────────────────────
    with st.expander("📋 Dataset & Run Info"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples",   len(y))
        col2.metric("Features Used",   X.shape[1])
        col3.metric("Classes",         len(np.unique(y)))
        col4.metric("Test Size",       f"{int(test_size * 100)}%")
        col1.metric("Train Samples",   len(X_train))
        col2.metric("Test Samples",    len(X_test))
        col3.metric("Random Seed",     int(random_seed))
        col4.metric("Feature Scaling", "Yes" if scale_data else "No")

        st.markdown("**Class Distribution:**")
        class_dist = pd.Series(Counter(y)).sort_index()
        st.bar_chart(class_dist)

    # ─── Results Table ─────────────────────────────────────────────────────────
    st.subheader("📊 Results Table")
    rows = []
    for i, level in enumerate(distortion_levels):
        row = {"Level": round(level, 3)}
        for name in selected_models:
            short = "".join([w[0] for w in name.split()])  # RF, LR, SVM, DT, KNN, GB
            row[f"{short} Acc"] = round(all_results[name][i]["accuracy"],   3)
            row[f"{short} F1"]  = round(all_results[name][i]["f1"],         3)
            row[f"{short} AUC"] = round(all_results[name][i]["roc_auc"],    3)
            row[f"{short} Conf"]= round(all_results[name][i]["confidence"], 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True)

    # ─── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv"
    )

    # ─── Save Run ──────────────────────────────────────────────────────────────
    settings = {
        "max_distortion_level": max_level,
        "num_levels":           num_levels,
        "test_size":            test_size,
        "scale_data":           scale_data,
        "random_seed":          int(random_seed),
        "selected_models":      selected_models,
    }
    distortions_used = {
        "gaussian_noise":     use_noise,
        "covariate_drift":    use_drift,
        "distribution_shift": use_dist_shift,
        "class_imbalance":    use_imbalance,
        "missing_values":     use_missing,
        "label_noise":        use_label_noise,
        "outlier_injection":  use_outliers,
        "feature_corruption": use_corruption,
    }

    # Convert all_results for JSON serialization
    rf_res  = all_results.get("Random Forest",       [empty_result] * len(distortion_levels))
    lr_res  = all_results.get("Logistic Regression", [empty_result] * len(distortion_levels))
    svm_res = all_results.get("SVM",                 [empty_result] * len(distortion_levels))

    run_id, run_dir = save_run(
        ds_name, settings, distortions_used,
        distortion_levels, rf_res, lr_res, svm_res, fig
    )
    st.info(f"💾 Run #{run_id:03d} saved to `{run_dir}/`")

# ─── Run History ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🕓 Run History")
all_runs = load_all_runs()
if not all_runs:
    st.info("No runs saved yet. Run a simulation to see history here.")
else:
    st.markdown(f"**{len(all_runs)} run(s) saved**")
    for run in reversed(all_runs):
        distortions_on = [k for k, v in run["distortions_used"].items() if v]
        models_used    = run["settings"].get("selected_models", ["Random Forest", "Logistic Regression", "SVM"])
        with st.expander(
            f"Run #{run['run_id']:03d} — {run['timestamp']} — "
            f"Dataset: {run['dataset']} — "
            f"Models: {', '.join(models_used)}"
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dataset",   run["dataset"])
            col2.metric("Max Level", run["settings"]["max_distortion_level"])
            col3.metric("Test Size", f"{int(run['settings']['test_size']*100)}%")
            col4.metric("Seed",      run["settings"]["random_seed"])

            st.caption(
                f"Distortions: {', '.join(distortions_on) if distortions_on else 'None'}"
            )

            if run.get("_png_path") and os.path.exists(run["_png_path"]):
                st.image(run["_png_path"], use_column_width=True)
            else:
                st.warning("Chart image not found for this run.")

            rows = []
            levels_list = run["levels"]
            rf_res  = run["results"]["random_forest"]
            lr_res  = run["results"]["logistic_regression"]
            svm_res = run["results"]["svm"]
            for i, level in enumerate(levels_list):
                rows.append({
                    "Level":   round(level, 3),
                    "RF Acc":  round(rf_res[i].get("accuracy",  0), 3),
                    "RF F1":   round(rf_res[i].get("f1",        0), 3),
                    "RF AUC":  round(rf_res[i].get("roc_auc",   0), 3),
                    "LR Acc":  round(lr_res[i].get("accuracy",  0), 3),
                    "LR F1":   round(lr_res[i].get("f1",        0), 3),
                    "LR AUC":  round(lr_res[i].get("roc_auc",   0), 3),
                    "SVM Acc": round(svm_res[i].get("accuracy", 0), 3),
                    "SVM F1":  round(svm_res[i].get("f1",       0), 3),
                    "SVM AUC": round(svm_res[i].get("roc_auc",  0), 3),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
