import numpy as np
from distortions import add_noise, add_drift, add_distribution_shift, add_class_imbalance, apply_distortion
from evaluation import evaluate
from model import load_dataset, train_models
from sklearn.model_selection import train_test_split


def test_add_noise_shape():
    X = np.random.rand(30, 4)
    assert add_noise(X, 0.1).shape == X.shape


def test_add_drift_shape():
    X = np.random.rand(30, 4)
    assert add_drift(X, 0.1).shape == X.shape


def test_add_distribution_shift_shape():
    X = np.random.rand(30, 4)
    assert add_distribution_shift(X, 0.1).shape == X.shape


def test_class_imbalance_reduces_minority():
    X = np.random.rand(90, 4)
    y = np.array([0]*30 + [1]*30 + [2]*30)
    X_new, y_new = add_class_imbalance(X, y, level=0.5)
    assert np.sum(y_new == 0) >= np.sum(y_new == 1)


def test_apply_distortion_returns_arrays():
    X = np.random.rand(30, 4)
    y = np.array([0]*10 + [1]*10 + [2]*10)
    X_d, y_d = apply_distortion(X, y, 0.1, 0.1, 0.1, 0.1)
    assert isinstance(X_d, np.ndarray)
    assert isinstance(y_d, np.ndarray)


def test_evaluate_returns_dict():
    X, y, _ = load_dataset("iris")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, lr, svm = train_models(X_train, y_train)
    result = evaluate(rf, X_test, y_test)
    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_all_metrics_between_0_and_1():
    X, y, _ = load_dataset("iris")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, _, _ = train_models(X_train, y_train)
    result = evaluate(rf, X_test, y_test)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"
