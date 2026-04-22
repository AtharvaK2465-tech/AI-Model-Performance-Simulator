import numpy as np


def add_noise(X, level):
    """Gaussian noise on all features."""
    noise = np.random.randn(*X.shape) * level
    return X + noise


def add_drift(X, level):
    """Mean shift (covariate drift) — shifts feature distributions."""
    drift = np.ones(X.shape[1]) * level * 2  # directional, not symmetric
    return X + drift


def add_distribution_shift(X, level):
    """
    Scales feature variance to simulate distribution shift.
    level=0 → no change, level=0.4 → heavy compression/expansion of spread.
    """
    scale_factor = 1 + level * 3
    mean = np.mean(X, axis=0)
    return mean + (X - mean) * scale_factor


def add_class_imbalance(X, y, level):
    """
    Drops samples of minority classes proportionally to level.
    level=0 → balanced, level=0.4 → heavy class imbalance.
    """
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


def apply_distortion(X, y, noise_level, drift_level, dist_shift_level, imbalance_level):
    """Apply all four distortion types in sequence."""
    X = add_noise(X, noise_level)
    X = add_drift(X, drift_level)
    X = add_distribution_shift(X, dist_shift_level)
    X, y = add_class_imbalance(X, y, imbalance_level)
    return X, y
