import numpy as np

# Noise
def add_noise(X, level):
    noise = np.random.randn(*X.shape) * level
    return X + noise

# Data Drift
def add_drift(X, level):
    drift = np.random.normal(0, level, X.shape)
    return X + drift

# Optional: combine both
def apply_distortion(X, noise_level, drift_level):
    X_noisy = add_noise(X, noise_level)
    X_drifted = add_drift(X_noisy, drift_level)
    return X_drifted