from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average='macro', zero_division=0),
        "recall":    recall_score(y_test, pred, average='macro', zero_division=0),
        "f1":        f1_score(y_test, pred, average='macro', zero_division=0),
    }
