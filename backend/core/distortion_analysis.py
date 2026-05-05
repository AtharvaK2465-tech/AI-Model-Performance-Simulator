import numpy as np
from backend.core.distortions import apply_distortion
from backend.core.evaluation import evaluate


DISTORTION_NAMES = [
    "Gaussian Noise",
    "Covariate Drift",
    "Distribution Shift",
    "Class Imbalance",
    "Missing Values",
    "Label Noise",
    "Outlier Injection",
    "Feature Corruption",
]

DISTORTION_KEYS = [
    "noise_level",
    "drift_level",
    "dist_shift_level",
    "imbalance_level",
    "missing_level",
    "label_noise_level",
    "outlier_level",
    "corruption_level",
]


def run_per_distortion_analysis(trained_models, X_test, y_test, level=0.3):
    """
    For each distortion type, apply it in isolation at `level`
    and evaluate all models. Returns dict:
    {
        distortion_name: {model_name: metrics_dict}
    }
    """
    results = {}

    for d_name, d_key in zip(DISTORTION_NAMES, DISTORTION_KEYS):
        kwargs = {k: 0.0 for k in DISTORTION_KEYS}
        kwargs[d_key] = level

        try:
            X_dist, y_dist = apply_distortion(
                X_test.copy(), y_test.copy(), **kwargs
            )

            if len(X_dist) == 0 or len(np.unique(y_dist)) < 2:
                results[d_name] = {
                    name: {"accuracy": 0, "f1": 0, "roc_auc": 0, "confidence": 0}
                    for name in trained_models
                }
            else:
                results[d_name] = {}
                for name, model in trained_models.items():
                    results[d_name][name] = evaluate(model, X_dist, y_dist)

        except Exception:
            results[d_name] = {
                name: {"accuracy": 0, "f1": 0, "roc_auc": 0, "confidence": 0}
                for name in trained_models
            }

    return results
