import numpy as np


def add_noise(X, level):
    """Gaussian noise on all features."""
    noise = np.random.randn(*X.shape) * level
    return X + noise


def add_drift(X, level):
    """Mean shift (covariate drift) — shifts feature distributions."""
    drift = np.ones(X.shape[1]) * level * 2
    return X + drift


def add_distribution_shift(X, level):
    """Scales feature variance to simulate distribution shift."""
    scale_factor = 1 + level * 3
    mean = np.mean(X, axis=0)
    return mean + (X - mean) * scale_factor


def add_class_imbalance(X, y, level):
    """Drops samples of minority classes proportionally to level."""
    classes = np.unique(y)
    majority = classes[np.argmax([np.sum(y == c) for c in classes])]
    X_new, y_new = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if c == majority:
            X_new.append(X[idx])
            y_new.append(y[idx])
        else:
            keep = max(1, int(len(idx) * (1 - level)))
            chosen = np.random.choice(idx, keep, replace=False)
            X_new.append(X[chosen])
            y_new.append(y[chosen])
    return np.vstack(X_new), np.concatenate(y_new)


def add_missing_values(X, level):
    """
    Randomly injects NaN into features proportional to level.
    level=0.1 → 10% of values become NaN, filled with column mean.
    """
    X = X.copy().astype(float)
    mask = np.random.rand(*X.shape) < level
    X[mask] = np.nan
    # Fill NaN with column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_idx = np.isnan(X[:, j])
        if nan_idx.any():
            X[nan_idx, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0
    return X


def add_label_noise(y, level):
    """
    Randomly flips class labels proportional to level.
    level=0.1 → 10% of labels randomly reassigned.
    Simulates annotation errors.
    """
    y = y.copy()
    classes = np.unique(y)
    n_flip  = int(len(y) * level)
    if n_flip == 0:
        return y
    flip_idx = np.random.choice(len(y), n_flip, replace=False)
    for idx in flip_idx:
        other_classes = classes[classes != y[idx]]
        if len(other_classes) > 0:
            y[idx] = np.random.choice(other_classes)
    return y


def add_outliers(X, level):
    """
    Injects extreme outlier values into random samples.
    level=0.1 → 10% of samples get outlier values (±5 std devs).
    """
    X      = X.copy()
    n_out  = max(1, int(len(X) * level))
    std    = np.std(X, axis=0) + 1e-8
    mean   = np.mean(X, axis=0)
    idx    = np.random.choice(len(X), n_out, replace=False)
    signs  = np.random.choice([-1, 1], size=(n_out, X.shape[1]))
    X[idx] = mean + signs * std * 5
    return X


def add_feature_corruption(X, level):
    """
    Zeros out random feature columns proportional to level.
    level=0.1 → 10% of feature columns set to 0.
    Simulates sensor failure or missing modality.
    """
    X          = X.copy()
    n_corrupt  = max(1, int(X.shape[1] * level))
    cols       = np.random.choice(X.shape[1], n_corrupt, replace=False)
    X[:, cols] = 0.0
    return X


def apply_distortion(
    X, y,
    noise_level       = 0.0,
    drift_level       = 0.0,
    dist_shift_level  = 0.0,
    imbalance_level   = 0.0,
    missing_level     = 0.0,
    label_noise_level = 0.0,
    outlier_level     = 0.0,
    corruption_level  = 0.0,
):
    """Apply selected distortion types in sequence."""
    if noise_level       > 0: X          = add_noise(X, noise_level)
    if drift_level       > 0: X          = add_drift(X, drift_level)
    if dist_shift_level  > 0: X          = add_distribution_shift(X, dist_shift_level)
    if missing_level     > 0: X          = add_missing_values(X, missing_level)
    if outlier_level     > 0: X          = add_outliers(X, outlier_level)
    if corruption_level  > 0: X          = add_feature_corruption(X, corruption_level)
    if label_noise_level > 0: y          = add_label_noise(y, label_noise_level)
    if imbalance_level   > 0: X, y       = add_class_imbalance(X, y, imbalance_level)
    return X, y
