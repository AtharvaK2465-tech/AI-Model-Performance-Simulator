# 🤖 AI Model Performance Simulator

Simulate how ML classification models degrade under real-world data distortions — noise, drift, imbalance, missing values, label noise, outliers, and more.

![results](results.png)

![per_distortion](per_distortion.png)

## What it does

Trains **Random Forest**, **Logistic Regression**, and **SVM** on clean data, then evaluates all three across increasing distortion levels. Tracks **6 metrics** at each level and produces a **2×3 comparison chart**.

## Models

| Model | Details |
|---|---|
| Random Forest | 100 estimators |
| Logistic Regression | max_iter=2000 |
| SVM | RBF kernel |

## Metrics Tracked

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision (macro) | Avg precision across all classes |
| Recall (macro) | Avg recall across all classes |
| F1 Score (macro) | Harmonic mean of precision & recall |
| ROC-AUC Score | Discrimination ability across classes |
| Model Confidence | Mean max prediction probability |

## Distortion Types

| Type | Description |
|---|---|
| Gaussian Noise | Random noise added to all features |
| Covariate Drift | Mean shift in feature values |
| Distribution Shift | Feature variance expansion/compression |
| Class Imbalance | Minority classes progressively dropped |
| Missing Values | Random NaN injection, filled with column mean |
| Label Noise | Random class label flipping (annotation errors) |
| Outlier Injection | Extreme values (±5 std devs) in random samples |
| Feature Corruption | Random feature columns zeroed out (sensor failure) |

## Dataset Support

- **Built-in:** Iris, Wine, Breast Cancer (sklearn)
- **Custom CSV:** Upload any CSV — auto-detects numeric features, auto-encodes categorical columns, drops ID-like and high-cardinality columns

## Simulation Controls

| Control | Description |
|---|---|
| Max Distortion Level | How severe distortions get (0.1 mild → 1.0 extreme) |
| Number of Distortion Levels | Steps between 0 and max level |
| Test Set Size | Fraction of data held out for testing |
| Feature Scaling | StandardScaler on/off |
| Random Seed | Reproducible train/test splits |
| Distortion Type Checkboxes | Choose which distortions to apply |

## Setup

```bash
git clone https://github.com/n-a-n-d-a-n/AI-Model-Performance-Simulator.git
cd AI-Model-Performance-Simulator
pip install -r requirements.txt
```

## Usage

**CLI (terminal):**
```bash
# Default (Iris dataset)
python main.py

# Different dataset
python main.py --dataset wine
python main.py --dataset breast_cancer

# Custom distortion levels
python main.py --levels 0 0.05 0.1 0.2 0.3 0.5

# Custom CSV
python main.py --csv path/to/your/data.csv

# Save plot
python main.py --save output/results.png
```

**Web Dashboard (Streamlit):**
```bash
streamlit run app.py
```

## Run Tests

```bash
pytest test_simulator.py -v
```

## Project Structure

```
├── app.py                # Streamlit web dashboard
├── main.py               # CLI entry point
├── model.py              # Model training + dataset loading
├── distortions.py        # All 8 distortion functions
├── evaluation.py         # 6 metrics including ROC-AUC + confidence
├── visualization.py      # 2×3 metrics chart
├── test_simulator.py     # Unit tests (7/7 passing)
├── requirements.txt
└── .gitignore
```

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
streamlit
```
