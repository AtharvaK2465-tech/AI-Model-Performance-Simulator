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
from distortion_analysis import run_per_distortion_analysis, DISTORTION_NAMES
from reliability_analysis import (
    compute_reliability_horizons,
    plot_reliability_windows,
    get_reliability_verdict,
)
from confusion_matrix_analysis import compute_confusion_matrices, plot_confusion_matrices
from degradation_analysis import compute_degradation, plot_degradation, get_robustness_tier

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Model Performance Simulator", layout="wide")
st.title("AI Model Performance Simulator")
st.markdown("Simulate how ML models degrade under real-world data distortions.")

# ─── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "sim_done":             False,
    "rel_df":               None,
    "all_results":          None,
    "distortion_levels":    None,
    "selected_models_snap": None,
    "ds_name_snap":         None,
    "max_level_snap":       None,
    "best_baseline":        0.75,
    "suggested_threshold":  0.75,
    "fig_main":             None,
    "fig_analysis":         None,
    "analysis_df":          None,
    "analysis_results":     None,
    "settings":             None,
    "distortions_used":     None,
    "rf_res":               None,
    "lr_res":               None,
    "svm_res":              None,
    "run_saved":            False,
    "last_run_id":          None,
    "last_run_dir":         None,
    "trained_models":       None,
    "X_test_clean":         None,
    "X_test_max_dist":      None,
    "y_test":               None,
    "y_test_max_dist":      None,
    "fig_confusion":        None,
    "fig_degradation":      None,
    "summary_df":           None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helper: build per-distortion bar chart ────────────────────────────────────
def _build_distortion_bar_chart(analysis_results, metric_choice,
                                 selected_models, max_level, ds_name):
    fig_a, ax_a = plt.subplots(figsize=(14, 6))
    x      = np.arange(len(DISTORTION_NAMES))
    width  = 0.8 / max(len(selected_models), 1)
    offset = -(len(selected_models) - 1) / 2

    for i, name in enumerate(selected_models):
        vals = [
            analysis_results[d][name].get(metric_choice, 0)
            for d in DISTORTION_NAMES
        ]
        ax_a.bar(
            x + (offset + i) * width, vals,
            width=width, label=name,
            color=MODEL_COLORS.get(name, "gray"), alpha=0.85
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(DISTORTION_NAMES, rotation=25, ha='right', fontsize=9)
    ax_a.set_ylabel(metric_choice.replace("_", " ").capitalize())
    ax_a.set_title(
        f"Model {metric_choice.capitalize()} Under Each Distortion Type "
        f"(level={max_level})\nDataset: {ds_name}",
        fontsize=12, fontweight="bold"
    )
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(fontsize=8)
    ax_a.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig_a

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
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
        st.sidebar.success(f"Loaded: {dataset_name}")
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
                st.error("Uploaded CSV is empty after cleaning.")
                st.stop()
            if len(df) < 20:
                st.error(f"Dataset has only {len(df)} rows. Need at least 20 samples.")
                st.stop()
            if len(df.columns) < 2:
                st.error("Dataset needs at least 2 columns (1 feature + 1 target).")
                st.stop()

            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

            columns    = df.columns.tolist()
            target_col = st.sidebar.selectbox(
                "Select Target Column", columns, index=len(columns) - 1
            )

            if is_regression_target(df[target_col].values):
                st.warning(
                    f"The column **'{target_col}'** looks like a continuous/regression target "
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
                st.sidebar.warning(f"Dropped ID-like columns: {id_like_cols}")

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
                    st.sidebar.info(f"Auto-encoded: {encoded_cols}")
                if dropped_cols:
                    st.sidebar.warning(f"Dropped high-cardinality columns: {dropped_cols}")

            all_feature_cols = numeric_cols + encoded_cols

            if len(all_feature_cols) == 0:
                st.error("No usable feature columns found.")
                st.stop()
            if len(all_feature_cols) == 1:
                st.warning("Only 1 feature column found. Results may be unreliable.")

            feature_df   = feature_df[all_feature_cols].copy()
            before_dedup = len(feature_df)
            feature_df   = feature_df.drop_duplicates()
            dupes        = before_dedup - len(feature_df)
            if dupes > 0:
                st.sidebar.info(f"Removed {dupes} duplicate rows.")

            before_nan  = len(feature_df)
            feature_df.dropna(inplace=True)
            dropped_nan = before_nan - len(feature_df)
            if dropped_nan > 0:
                st.sidebar.info(f"Dropped {dropped_nan} rows with missing values.")

            if len(feature_df) < 20:
                st.error(f"Only {len(feature_df)} usable rows remain. Need at least 20.")
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
                st.sidebar.info(f"Dropped {count} rows with missing target values.")
                X     = X[~target_nan_mask]
                y_raw = y_raw[~target_nan_mask]

            le = LabelEncoder()
            try:
                y = le.fit_transform(y_raw.astype(str))
            except Exception as enc_err:
                st.error(f"Could not encode target column: {enc_err}")
                st.stop()

            n_classes = len(np.unique(y))
            n_samples = len(y)

            if n_samples < 20:
                st.error(f"Too few samples ({n_samples}). Need at least 20.")
                st.stop()
            if n_classes < 2:
                st.error(f"Target column has only 1 class. Need at least 2.")
                st.stop()
            if n_classes > 50:
                st.error(f"Target column has {n_classes} classes - looks like regression/ID.")
                st.stop()

            class_counts    = Counter(y)
            min_class_count = min(class_counts.values())
            max_class_count = max(class_counts.values())
            imbalance_ratio = max_class_count / min_class_count

            if min_class_count < 2:
                st.error(f"At least one class has only {min_class_count} sample(s).")
                st.stop()
            if imbalance_ratio > 10:
                st.warning(f"Dataset is highly imbalanced (ratio {imbalance_ratio:.1f}x).")

            ds_name = uploaded_file.name
            st.sidebar.success(f"CSV loaded: {ds_name}")
            st.sidebar.markdown(f"**Samples:** {n_samples}")
            st.sidebar.markdown(
                f"**Classes:** {n_classes} -> {list(le.classes_[:5])}"
                + ("..." if n_classes > 5 else "")
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
            st.stop()
        except pd.errors.ParserError:
            st.error("Could not parse CSV.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error reading CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV file from the sidebar, or switch to a built-in dataset.")

# ─── Model Selection ───────────────────────────────────────────────────────────
st.sidebar.header("Model Selection")
st.sidebar.markdown("Select models to compare:")

model_selections = {}
for model_name in ALL_MODELS.keys():
    default = model_name in ["Random Forest", "Logistic Regression", "SVM"]
    model_selections[model_name] = st.sidebar.checkbox(model_name, value=default)

selected_models = [k for k, v in model_selections.items() if v]

if len(selected_models) == 0:
    st.sidebar.warning("No models selected. Please select at least one.")
elif len(selected_models) == 1:
    st.sidebar.info("Select at least 2 models for a meaningful comparison.")

# ─── Simulation Settings ───────────────────────────────────────────────────────
st.sidebar.header("Simulation Settings")

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
st.sidebar.header("Distortion Types")
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
    st.sidebar.warning("No distortions selected. Results will be flat lines.")

# ─── Run Simulation ────────────────────────────────────────────────────────────
if st.sidebar.button("Run Simulation"):

    st.session_state.run_saved = False

    if X is None or y is None:
        st.warning("Please select or upload a valid dataset first.")
        st.stop()

    if len(selected_models) == 0:
        st.warning("Please select at least one model.")
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
            st.error(f"Train/test split failed: {ve}")
            st.stop()
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    st.success(
        f"Models trained on **{ds_name}** | "
        f"Samples: {len(y)} | Features: {X.shape[1]} | "
        f"Classes: {len(np.unique(y))} | "
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Models: {', '.join(selected_models)}"
    )

    empty_result = {
        "accuracy": 0, "precision": 0, "recall": 0,
        "f1": 0, "roc_auc": 0, "confidence": 0
    }

    all_results = {name: [] for name in selected_models}
    progress    = st.progress(0)
    status      = st.empty()

    for i, level in enumerate(distortion_levels):
        status.text(f"Evaluating distortion level {round(level, 3)}...")
        try:
            X_dist, y_dist = apply_distortion(
                X_test.copy(), y_test.copy(),
                noise_level       = level if use_noise       else 0.0,
                drift_level       = level if use_drift       else 0.0,
                dist_shift_level  = level if use_dist_shift  else 0.0,
                imbalance_level   = level if use_imbalance   else 0.0,
                missing_level     = level if use_missing      else 0.0,
                label_noise_level = level if use_label_noise else 0.0,
                outlier_level     = level if use_outliers    else 0.0,
                corruption_level  = level if use_corruption  else 0.0,
            )

            if len(X_dist) == 0 or len(np.unique(y_dist)) < 2:
                st.warning(f"Level {round(level, 2)}: Not enough class diversity - skipping.")
                for name in selected_models:
                    all_results[name].append(empty_result.copy())
            else:
                for name, model in trained_models.items():
                    all_results[name].append(evaluate(model, X_dist, y_dist))

        except Exception as e:
            st.warning(f"Error at level {round(level, 2)}: {e} - skipping.")
            for name in selected_models:
                all_results[name].append(empty_result.copy())

        progress.progress((i + 1) / len(distortion_levels))

    status.text("Simulation complete!")

    # ─── Plot 2x3 grid ─────────────────────────────────────────────────────────
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
    with st.expander("Dataset & Run Info"):
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
    st.subheader("Results Table")
    rows = []
    for i, level in enumerate(distortion_levels):
        row = {"Level": round(level, 3)}
        for name in selected_models:
            short = "".join([w[0] for w in name.split()])
            row[f"{short} Acc"]  = round(all_results[name][i]["accuracy"],   3)
            row[f"{short} F1"]   = round(all_results[name][i]["f1"],         3)
            row[f"{short} AUC"]  = round(all_results[name][i]["roc_auc"],    3)
            row[f"{short} Conf"] = round(all_results[name][i]["confidence"], 3)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv"
    )

    # ─── Build max-distortion test set for confusion matrix ───────────────────
    X_test_max_dist = X_test.copy()
    y_test_max_dist = y_test.copy()
    X_test_max_dist, y_test_max_dist = apply_distortion(
        X_test_max_dist, y_test_max_dist,
        noise_level       = max_level if use_noise       else 0.0,
        drift_level       = max_level if use_drift       else 0.0,
        dist_shift_level  = max_level if use_dist_shift  else 0.0,
        imbalance_level   = max_level if use_imbalance   else 0.0,
        missing_level     = max_level if use_missing      else 0.0,
        label_noise_level = max_level if use_label_noise else 0.0,
        outlier_level     = max_level if use_outliers    else 0.0,
        corruption_level  = max_level if use_corruption  else 0.0,
    )

    # ─── Per-Distortion Analysis (compute only) ────────────────────────────────
    with st.spinner("Running per-distortion analysis..."):
        analysis_results = run_per_distortion_analysis(
            trained_models, X_test, y_test, level=max_level
        )

    rows_a = []
    for d_name in DISTORTION_NAMES:
        row = {"Distortion": d_name}
        for name in selected_models:
            short = "".join([w[0] for w in name.split()])
            row[f"{short} Acc"] = round(analysis_results[d_name][name].get("accuracy", 0), 3)
            row[f"{short} F1"]  = round(analysis_results[d_name][name].get("f1",       0), 3)
            row[f"{short} AUC"] = round(analysis_results[d_name][name].get("roc_auc",  0), 3)
        rows_a.append(row)
    analysis_df = pd.DataFrame(rows_a)

    # ─── Build long-form rel_df ────────────────────────────────────────────────
    rel_rows = []
    for name in selected_models:
        for i, level in enumerate(distortion_levels):
            rel_rows.append({
                "model_name":       name,
                "distortion_level": level,
                "accuracy":         all_results[name][i]["accuracy"],
                "precision":        all_results[name][i]["precision"],
                "recall":           all_results[name][i]["recall"],
                "f1":               all_results[name][i]["f1"],
                "roc_auc":          all_results[name][i]["roc_auc"],
                "confidence":       all_results[name][i]["confidence"],
            })
    rel_df = pd.DataFrame(rel_rows)

    # Auto-suggest threshold
    baseline_accs = [all_results[name][0]["accuracy"] for name in selected_models]
    best_baseline = max(baseline_accs) if baseline_accs else 0.75
    suggested     = max(0.40, min(best_baseline * 0.80, 0.95))
    suggested     = round(round(suggested / 0.05) * 0.05, 2)

    # Pre-compute degradation summary
    summary_df = compute_degradation(rel_df, metric="accuracy")

    # ─── Persist everything to session_state ──────────────────────────────────
    st.session_state.sim_done              = True
    st.session_state.rel_df               = rel_df
    st.session_state.all_results          = all_results
    st.session_state.distortion_levels    = distortion_levels
    st.session_state.selected_models_snap = selected_models
    st.session_state.ds_name_snap         = ds_name
    st.session_state.max_level_snap       = max_level
    st.session_state.suggested_threshold  = suggested
    st.session_state.best_baseline        = best_baseline
    st.session_state.fig_main             = fig
    st.session_state.analysis_df          = analysis_df
    st.session_state.analysis_results     = analysis_results
    st.session_state.trained_models       = trained_models
    st.session_state.X_test_clean         = X_test
    st.session_state.X_test_max_dist      = X_test_max_dist
    st.session_state.y_test               = y_test
    st.session_state.y_test_max_dist      = y_test_max_dist
    st.session_state.summary_df           = summary_df
    st.session_state.settings             = {
        "max_distortion_level": max_level,
        "num_levels":           num_levels,
        "test_size":            test_size,
        "scale_data":           scale_data,
        "random_seed":          int(random_seed),
        "selected_models":      selected_models,
    }
    st.session_state.distortions_used = {
        "gaussian_noise":     use_noise,
        "covariate_drift":    use_drift,
        "distribution_shift": use_dist_shift,
        "class_imbalance":    use_imbalance,
        "missing_values":     use_missing,
        "label_noise":        use_label_noise,
        "outlier_injection":  use_outliers,
        "feature_corruption": use_corruption,
    }
    st.session_state.rf_res  = all_results.get("Random Forest",
                               [empty_result] * len(distortion_levels))
    st.session_state.lr_res  = all_results.get("Logistic Regression",
                               [empty_result] * len(distortion_levels))
    st.session_state.svm_res = all_results.get("SVM",
                               [empty_result] * len(distortion_levels))

# ─── Per-Distortion Analysis (outside button block) ────────────────────────────
if st.session_state.sim_done and st.session_state.analysis_results is not None:

    st.markdown("---")
    st.subheader("Per-Distortion Analysis")
    st.markdown(
        f"Each distortion type applied **individually** at level "
        f"`{st.session_state.max_level_snap}` "
        "-- shows which one hurts each model the most."
    )

    metric_choice = st.selectbox(
        "Metric to visualize", ["accuracy", "f1", "roc_auc"], index=0,
        key="metric_choice"
    )

    fig_a = _build_distortion_bar_chart(
        st.session_state.analysis_results,
        metric_choice,
        st.session_state.selected_models_snap,
        st.session_state.max_level_snap,
        st.session_state.ds_name_snap,
    )
    st.pyplot(fig_a)
    plt.close(fig_a)
    st.session_state.fig_analysis = fig_a

    st.subheader("Per-Distortion Results Table")
    st.dataframe(st.session_state.analysis_df, use_container_width=True)

    st.download_button(
        label="Download Analysis as CSV",
        data=st.session_state.analysis_df.to_csv(index=False),
        file_name="per_distortion_analysis.csv",
        mime="text/csv"
    )

# ─── Degradation Rate Analysis (outside button block) ──────────────────────────
if st.session_state.sim_done and st.session_state.rel_df is not None:

    st.markdown("---")
    st.subheader("Degradation Rate Analysis")

    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Shows how fast each model degrades as distortion increases. "
            "**Robustness Score** is the area under the performance curve normalised to 0-100 "
            "-- higher means the model held up better across all distortion levels. "
            "**Deg Rate/Level** is the metric value lost per unit of distortion."
        )

    deg_metric = st.selectbox(
        "Metric for degradation analysis",
        ["accuracy", "f1", "precision", "recall"],
        index=0,
        key="deg_metric"
    )

    summary_df = compute_degradation(st.session_state.rel_df, metric=deg_metric)
    st.session_state.summary_df = summary_df

    deg_fig = plot_degradation(
        st.session_state.rel_df, summary_df, MODEL_COLORS, metric=deg_metric
    )
    st.pyplot(deg_fig, use_container_width=True)
    st.session_state.fig_degradation = deg_fig
    plt.close(deg_fig)

    # ── Robustness summary table ───────────────────────────────────────────────
    st.markdown("**Robustness Summary Table**")

    def _colour_robustness(val):
        if not isinstance(val, (int, float)):
            return ""
        if val >= 90:
            return "background-color: #d4edda; color: #155724"
        elif val >= 75:
            return "background-color: #fff3cd; color: #856404"
        elif val >= 55:
            return "background-color: #ffe0b2; color: #6d4c00"
        else:
            return "background-color: #f8d7da; color: #721c24"

    styled_summary = (
        summary_df.style
        .map(_colour_robustness, subset=["Robustness Score"])
        .format({"% Drop": "{:.2f}%", "Robustness Score": "{:.2f}"})
    )
    st.dataframe(styled_summary, use_container_width=True)

    # ── Tier badges ───────────────────────────────────────────────────────────
    st.markdown("**Robustness Tiers:**")
    tier_cols = st.columns(len(summary_df))
    for col, (_, row) in zip(tier_cols, summary_df.iterrows()):
        tier = get_robustness_tier(row["Robustness Score"])
        col.metric(
            label=row["Model"],
            value=tier,
            delta=f"-{row['% Drop']:.1f}% drop",
            delta_color="inverse"
        )

    # ── Download ──────────────────────────────────────────────────────────────
    deg_csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Degradation Summary CSV",
        deg_csv,
        file_name="degradation_summary.csv",
        mime="text/csv"
    )

# ─── Confusion Matrix (outside button block) ───────────────────────────────────
if st.session_state.sim_done and st.session_state.trained_models is not None:

    st.markdown("---")
    st.subheader("Confusion Matrices -- Clean Baseline vs Max Distortion")

    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Each model is shown two confusion matrices side by side -- "
            "**clean baseline** (left) and **max distortion** (right). "
            "Each cell shows the raw count and row-normalised percentage. "
            "Accuracy is shown below each matrix."
        )

    with st.spinner("Computing confusion matrices..."):
        cm_results = compute_confusion_matrices(
            trained_models   = st.session_state.trained_models,
            X_train          = None,
            y_train          = None,
            X_test_clean     = st.session_state.X_test_clean,
            y_test           = st.session_state.y_test,
            X_test_distorted = st.session_state.X_test_max_dist,
            y_test_distorted = st.session_state.y_test_max_dist,
        )

    for name, res in cm_results.items():
        if "error" in res:
            st.warning(f"Confusion matrix error for {name}: {res['error']}")

    cm_fig = plot_confusion_matrices(cm_results, MODEL_COLORS)
    st.pyplot(cm_fig, use_container_width=True)
    st.session_state.fig_confusion = cm_fig
    plt.close(cm_fig)

    # ── Accuracy summary cards ────────────────────────────────────────────────
    st.markdown("**Accuracy Summary: Clean vs Max Distortion**")
    valid = {k: v for k, v in cm_results.items() if "error" not in v}
    if valid:
        cols = st.columns(len(valid))
        for col, (name, data) in zip(cols, valid.items()):
            cm_clean  = data["clean"]
            cm_dist   = data["distorted"]
            acc_clean = np.trace(cm_clean) / (cm_clean.sum() + 1e-9)
            acc_dist  = np.trace(cm_dist)  / (cm_dist.sum()  + 1e-9)
            drop      = acc_clean - acc_dist
            col.metric(
                label=name,
                value=f"{acc_clean:.1%}",
                delta=f"-{drop:.1%} at max distortion",
                delta_color="inverse"
            )

# ─── Reliability Analysis (outside button block) ───────────────────────────────
if st.session_state.sim_done and st.session_state.rel_df is not None:

    rel_df        = st.session_state.rel_df
    best_baseline = st.session_state.best_baseline
    suggested     = st.session_state.suggested_threshold
    snap_max      = st.session_state.max_level_snap

    st.markdown("---")
    st.subheader("Model Reliability Horizon")

    with st.expander("What is this?", expanded=False):
        st.markdown(
            "Defines a **reliability threshold** and finds the exact distortion level "
            "where each model crosses it -- the **reliability horizon**.\n\n"
            "- Green zones = safe operating range\n"
            "- Red zones = model should not be trusted\n\n"
            "**Tip:** The threshold is auto-suggested at 80% of the best model's clean "
            "baseline. Adjust it to match your deployment requirements."
        )

    st.info(
        f"Best baseline across selected models: **{best_baseline:.3f}** | "
        f"Auto-suggested threshold: **{suggested:.2f}** (80% of baseline) | "
        "Adjust below as needed."
    )

    col_thresh, col_metric_rel = st.columns(2)

    worst_baseline = rel_df.groupby("model_name")["accuracy"].first().min()
    slider_min     = max(0.05, round(round(worst_baseline * 0.5 / 0.05) * 0.05, 2))

    reliability_threshold = col_thresh.slider(
        "Reliability threshold",
        min_value=float(slider_min),
        max_value=0.99,
        value=float(suggested),
        step=0.05,
        key="rel_threshold",
        help=(
            f"Minimum acceptable performance. "
            f"Auto-set to {suggested:.0%} based on best baseline ({best_baseline:.1%})."
        )
    )
    reliability_metric = col_metric_rel.selectbox(
        "Metric",
        ["accuracy", "f1", "precision", "recall"],
        index=0,
        key="rel_metric"
    )

    horizons_df = compute_reliability_horizons(
        rel_df,
        metric=reliability_metric,
        threshold=reliability_threshold,
    )

    rel_fig = plot_reliability_windows(
        rel_df, horizons_df, MODEL_COLORS,
        metric=reliability_metric,
        threshold=reliability_threshold,
    )
    st.pyplot(rel_fig, use_container_width=True)
    plt.close(rel_fig)

    st.markdown("**Reliability Horizon Table**")

    def _colour_reliable_range(val):
        if not isinstance(val, (int, float)):
            return ""
        if val >= 90:
            return "background-color: #d4edda; color: #155724"
        elif val >= 60:
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #f8d7da; color: #721c24"

    styled_horizons = (
        horizons_df.style
        .map(_colour_reliable_range, subset=["Reliable Range (%)"])
        .format({"Reliable Range (%)": "{:.1f}%", "Horizon Level": "{:.4f}"})
    )
    st.dataframe(styled_horizons, use_container_width=True)

    # Best model banner
    if not horizons_df.empty:
        best = horizons_df.iloc[0]
        st.success(
            f"**Recommended Model: {best['Model']}** -- "
            f"Reliable for {best['Reliable Range (%)']:.1f}% of the distortion range "
            f"(until level `{best['Horizon Level']:.3f}`)"
        )

    verdict_md = get_reliability_verdict(horizons_df, snap_max)
    st.markdown(verdict_md)

    rel_csv = horizons_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Reliability Report CSV",
        rel_csv,
        file_name="reliability_horizons.csv",
        mime="text/csv",
    )

    # ─── Save Run (once per simulation) ───────────────────────────────────────
    if not st.session_state.run_saved:
        run_id, run_dir = save_run(
            st.session_state.ds_name_snap,
            st.session_state.settings,
            st.session_state.distortions_used,
            st.session_state.distortion_levels,
            st.session_state.rf_res,
            st.session_state.lr_res,
            st.session_state.svm_res,
            st.session_state.fig_main,
            fig_analysis=st.session_state.fig_analysis,
            analysis_df=st.session_state.analysis_df,
            fig_reliability=rel_fig,
            horizons_df=horizons_df,
            reliability_threshold=reliability_threshold,
            reliability_metric=reliability_metric,
            fig_confusion=st.session_state.fig_confusion,
            fig_degradation=st.session_state.fig_degradation,
            summary_df=st.session_state.summary_df,
        )
        st.session_state.run_saved    = True
        st.session_state.last_run_id  = run_id
        st.session_state.last_run_dir = str(run_dir)

    if st.session_state.last_run_id:
        st.info(
            f"Run #{st.session_state.last_run_id:03d} saved to "
            f"`{st.session_state.last_run_dir}/`"
        )

# ─── Run History ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Run History")
all_runs = load_all_runs()
if not all_runs:
    st.info("No runs saved yet. Run a simulation to see history here.")
else:
    st.markdown(f"**{len(all_runs)} run(s) saved**")
    for run in reversed(all_runs):
        distortions_on = [k for k, v in run["distortions_used"].items() if v]
        models_used    = run["settings"].get(
            "selected_models", ["Random Forest", "Logistic Regression", "SVM"]
        )
        with st.expander(
            f"Run #{run['run_id']:03d} -- {run['timestamp']} -- "
            f"Dataset: {run['dataset']} -- "
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
                st.markdown("**Main Simulation Chart:**")
                st.image(run["_png_path"], use_column_width=True)
            else:
                st.warning("Main chart not found for this run.")

            if run.get("_analysis_png_path") and os.path.exists(run["_analysis_png_path"]):
                st.markdown("**Per-Distortion Analysis Chart:**")
                st.image(run["_analysis_png_path"], use_column_width=True)

            if run.get("_degradation_png_path") and os.path.exists(run["_degradation_png_path"]):
                st.markdown("**Degradation Rate Analysis:**")
                st.image(run["_degradation_png_path"], use_column_width=True)

            if run.get("_confusion_png_path") and os.path.exists(run["_confusion_png_path"]):
                st.markdown("**Confusion Matrices:**")
                st.image(run["_confusion_png_path"], use_column_width=True)

            if run.get("_reliability_png_path") and os.path.exists(run["_reliability_png_path"]):
                st.markdown("**Reliability Horizon Chart:**")
                st.image(run["_reliability_png_path"], use_column_width=True)

            # Degradation summary from JSON
            deg_data = run.get("degradation", [])
            if deg_data:
                st.markdown("**Degradation Summary:**")
                deg_history = pd.DataFrame(deg_data)
                st.dataframe(deg_history, use_container_width=True)

            # Reliability summary from JSON
            rel_data = run.get("reliability", {})
            if rel_data and rel_data.get("horizons"):
                st.markdown(
                    f"**Reliability Settings:** "
                    f"Metric = `{rel_data.get('metric', 'N/A')}` | "
                    f"Threshold = `{rel_data.get('threshold', 'N/A')}`"
                )
                horizons_history = pd.DataFrame(rel_data["horizons"])
                st.markdown("**Reliability Horizon Table:**")
                st.dataframe(horizons_history, use_container_width=True)

                if (not horizons_history.empty
                        and "Reliable Range (%)" in horizons_history.columns):
                    best_row  = horizons_history.loc[
                        horizons_history["Reliable Range (%)"].idxmax()
                    ]
                    worst_row = horizons_history.loc[
                        horizons_history["Reliable Range (%)"].idxmin()
                    ]
                    rc1, rc2 = st.columns(2)
                    rc1.metric(
                        "Most Reliable Model",
                        best_row["Model"],
                        f"{best_row['Reliable Range (%)']:.1f}% of range"
                    )
                    rc2.metric(
                        "Least Reliable Model",
                        worst_row["Model"],
                        f"-{worst_row['Reliable Range (%)']:.1f}% of range",
                        delta_color="inverse"
                    )

            # Simulation results table
            rows      = []
            lvls      = run["levels"]
            rf_res_h  = run["results"]["random_forest"]
            lr_res_h  = run["results"]["logistic_regression"]
            svm_res_h = run["results"]["svm"]
            for i, level in enumerate(lvls):
                rows.append({
                    "Level":   round(level, 3),
                    "RF Acc":  round(rf_res_h[i].get("accuracy",  0), 3),
                    "RF F1":   round(rf_res_h[i].get("f1",        0), 3),
                    "RF AUC":  round(rf_res_h[i].get("roc_auc",   0), 3),
                    "LR Acc":  round(lr_res_h[i].get("accuracy",  0), 3),
                    "LR F1":   round(lr_res_h[i].get("f1",        0), 3),
                    "LR AUC":  round(lr_res_h[i].get("roc_auc",   0), 3),
                    "SVM Acc": round(svm_res_h[i].get("accuracy", 0), 3),
                    "SVM F1":  round(svm_res_h[i].get("f1",       0), 3),
                    "SVM AUC": round(svm_res_h[i].get("roc_auc",  0), 3),
                })
            st.markdown("**Simulation Results:**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
