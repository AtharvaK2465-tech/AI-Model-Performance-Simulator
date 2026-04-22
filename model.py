from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd
import numpy as np


DATASETS = {
    "iris":          load_iris,
    "wine":          load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name="iris", csv_path=None):
    """
    Returns (X, y, dataset_name).
    Pass csv_path to load a custom CSV (last column treated as target).
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values
        if y.dtype == object:
            classes = np.unique(y)
            mapping = {c: i for i, c in enumerate(classes)}
            y = np.array([mapping[v] for v in y])
        return X, y.astype(int), csv_path
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
    data = DATASETS[name]()
    return data.data, data.target, name


def train_models(X_train, y_train):
    """Returns trained (RandomForest, LogisticRegression, SVM)."""
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    lr  = LogisticRegression(max_iter=500, random_state=42)
    svm = SVC(kernel='rbf', random_state=42)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    return rf, lr, svm
