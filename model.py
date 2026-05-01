from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


DATASETS = {
    "iris":          load_iris,
    "wine":          load_wine,
    "breast_cancer": load_breast_cancer,
}

ALL_MODELS = {
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "SVM":                 lambda: SVC(kernel='rbf', random_state=42, probability=True),
    "Decision Tree":       lambda: DecisionTreeClassifier(random_state=42),
    "KNN":                 lambda: KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost":             lambda: XGBClassifier(n_estimators=100, random_state=42,
                                                  eval_metric='mlogloss',
                                                  verbosity=0),
}

MODEL_COLORS = {
    "Random Forest":       "steelblue",
    "Logistic Regression": "tomato",
    "SVM":                 "seagreen",
    "Decision Tree":       "orange",
    "KNN":                 "purple",
    "Gradient Boosting":   "crimson",
    "XGBoost":             "#e74c3c",
}

MODEL_MARKERS = {
    "Random Forest":       "o",
    "Logistic Regression": "s",
    "SVM":                 "^",
    "Decision Tree":       "D",
    "KNN":                 "P",
    "Gradient Boosting":   "*",
    "XGBoost":             "X",
}


def load_dataset(name="iris", csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        X  = df.iloc[:, :-1].values.astype(float)
        y  = df.iloc[:, -1].values
        if y.dtype == object:
            classes = np.unique(y)
            mapping = {c: i for i, c in enumerate(classes)}
            y = np.array([mapping[v] for v in y])
        return X, y.astype(int), csv_path
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")
    data = DATASETS[name]()
    return data.data, data.target, name


def train_models(X_train, y_train, selected_models=None):
    """
    Train selected models. Returns dict {model_name: trained_model}.
    If selected_models is None, trains all 7.
    """
    if selected_models is None:
        selected_models = list(ALL_MODELS.keys())

    trained = {}
    for name in selected_models:
        if name in ALL_MODELS:
            model = ALL_MODELS[name]()
            model.fit(X_train, y_train)
            trained[name] = model
    return trained
