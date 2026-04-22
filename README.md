# AI Model Performance Simulator

Simulates how ML classification models degrade under real-world data distortions — noise, drift, distribution shift, and class imbalance.
![results](results.png)

## What it does

Trains **Random Forest**, **Logistic Regression**, and **SVM** on clean data, then evaluates all three across increasing distortion levels. Tracks **Accuracy, Precision, Recall, and F1** at each level and produces a 2×2 comparison chart.

## Distortion types

| Type | Description |
|---|---|
| Gaussian Noise | Random noise added to all features |
| Covariate Drift | Mean shift in feature values |
| Distribution Shift | Feature variance expansion/compression |
| Class Imbalance | Minority classes progressively dropped |

## Setup

```bash
git clone https://github.com/AtharvaK2465-tech/AI-Model-Performance-Simulator.git
cd AI-Model-Performance-Simulator
pip install -r requirements.txt
```

## Usage

**Default (Iris dataset):**
```bash
python main.py
```

**Choose a different dataset:**
```bash
python main.py --dataset wine
python main.py --dataset breast_cancer
```

**Custom distortion levels:**
```bash
python main.py --levels 0 0.05 0.1 0.2 0.3 0.5
```

**Custom CSV file (last column = target label):**
```bash
python main.py --csv path/to/your/data.csv
```

**Save plot to a specific path:**
```bash
python main.py --save output/my_results.png
```

## Project structure

```
├── main.py           # Entry point & CLI
├── model.py          # Model training + dataset loading
├── distortions.py    # All 4 distortion functions
├── evaluation.py     # Metrics (accuracy, precision, recall, F1)
├── visualization.py  # 2x2 metrics chart
├── requirements.txt
└── .gitignore
```

## Models compared

- Random Forest (100 estimators)
- Logistic Regression
- SVM (RBF kernel)
