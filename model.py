from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):
    rf = RandomForestClassifier()
    lr = LogisticRegression(max_iter=200)

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return rf, lr