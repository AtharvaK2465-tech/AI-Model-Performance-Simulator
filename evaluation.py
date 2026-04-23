from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import numpy as np


def evaluate(model, X_test, y_test):
    pred  = model.predict(X_test)
    n_classes = len(np.unique(y_test))

    # ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, proba,
                    multi_class='ovr', average='macro'
                )
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    # Mean confidence (max probability per sample)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            confidence = float(np.mean(np.max(proba, axis=1)))
        else:
            confidence = 0.0
    except Exception:
        confidence = 0.0

    return {
        "accuracy":   accuracy_score(y_test, pred),
        "precision":  precision_score(y_test, pred, average='macro', zero_division=0),
        "recall":     recall_score(y_test, pred, average='macro', zero_division=0),
        "f1":         f1_score(y_test, pred, average='macro', zero_division=0),
        "roc_auc":    auc,
        "confidence": confidence,
    }
