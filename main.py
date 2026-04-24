import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from model import load_dataset, train_models
from distortions import apply_distortion
from evaluation import evaluate
from visualization import plot_results


def parse_args():
    parser = argparse.ArgumentParser(description="AI Model Performance Simulator")
    parser.add_argument("--dataset", default="iris",
                        choices=["iris", "wine", "breast_cancer"],
                        help="Built-in sklearn dataset to use")
    parser.add_argument("--csv",     default=None,
                        help="Path to a custom CSV file (last column = target)")
    parser.add_argument("--levels",  nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4],
                        help="Distortion levels to simulate")
    parser.add_argument("--save",    default="results.png",
                        help="Output path for the plot image")
    return parser.parse_args()


def main():
    args   = parse_args()
    levels = sorted(args.levels)

    X, y, ds_name = load_dataset(name=args.dataset, csv_path=args.csv)
    print(f"\n[✓] Dataset loaded: {ds_name}  | Samples: {len(y)}  | Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf, lr, svm = train_models(X_train, y_train)
    print("[✓] Models trained: Random Forest, Logistic Regression, SVM\n")

    rf_results, lr_results, svm_results = [], [], []

    header = f"{'Level':>7} | {'RF Acc':>7} {'RF F1':>7} {'RF AUC':>7} | {'LR Acc':>7} {'LR F1':>7} {'LR AUC':>7} | {'SVM Acc':>8} {'SVM F1':>7} {'SVM AUC':>8}"
    print(header)
    print("-" * len(header))

    for level in levels:
        X_dist, y_dist = apply_distortion(
            X_test.copy(), y_test.copy(),
            noise_level      = level,
            drift_level      = level,
            dist_shift_level = level,
            imbalance_level  = level
        )

        r_rf  = evaluate(rf,  X_dist, y_dist)
        r_lr  = evaluate(lr,  X_dist, y_dist)
        r_svm = evaluate(svm, X_dist, y_dist)

        rf_results.append(r_rf)
        lr_results.append(r_lr)
        svm_results.append(r_svm)

        print(
            f"{level:>7.1f} | "
            f"{r_rf['accuracy']:>7.3f} {r_rf['f1']:>7.3f} {r_rf['roc_auc']:>7.3f} | "
            f"{r_lr['accuracy']:>7.3f} {r_lr['f1']:>7.3f} {r_lr['roc_auc']:>7.3f} | "
            f"{r_svm['accuracy']:>8.3f} {r_svm['f1']:>7.3f} {r_svm['roc_auc']:>8.3f}"
        )

    plot_results(levels, rf_results, lr_results, svm_results, save_path=args.save)


if __name__ == "__main__":
    main()
