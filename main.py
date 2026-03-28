from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from model import train_models
from distortions import apply_distortion
from evaluation import evaluate
from visualization import plot_results

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
rf, lr = train_models(X_train, y_train)

# Distortion levels
levels = [0, 0.1, 0.2, 0.3, 0.4]

rf_acc = []
lr_acc = []

# Simulation loop
for level in levels:
    X_distorted = apply_distortion(X_test, level, level)

    acc_rf, _, _, _ = evaluate(rf, X_distorted, y_test)
    acc_lr, _, _, _ = evaluate(lr, X_distorted, y_test)

    rf_acc.append(acc_rf)
    lr_acc.append(acc_lr)

# Print results
print("\n--- Model Performance ---")
for i in range(len(levels)):
    print(f"Level {levels[i]} → RF: {rf_acc[i]:.2f}, LR: {lr_acc[i]:.2f}")

# Plot graph
plot_results(levels, rf_acc, lr_acc)