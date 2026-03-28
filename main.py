from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from model import train_model
from distortions import apply_distortion
from evaluation import evaluate
from visualization import plot_results

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = train_model(X_train, y_train)

# Simulation levels
levels = [0, 0.1, 0.2, 0.3, 0.4]
accuracies = []

for level in levels:
    X_distorted = apply_distortion(X_test, level, level)
    acc, _, _, _ = evaluate(model, X_distorted, y_test)
    accuracies.append(acc)

# Plot results
plot_results(levels, accuracies)