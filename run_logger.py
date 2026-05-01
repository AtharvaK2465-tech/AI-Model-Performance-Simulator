"""
run_logger.py
Saves every simulation run to results/run_NNN/ with:
  - run_NNN.json         (all metadata + results + reliability data)
  - results.png          (main 2x3 metrics chart)
  - per_distortion.png   (per-distortion bar chart)
  - reliability.png      (reliability horizon chart, if available)
"""

import os
import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def _get_next_run_id():
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    if not existing:
        return 1
    last = existing[-1].name  # e.g. "run_007"
    try:
        return int(last.split("_")[1]) + 1
    except Exception:
        return len(existing) + 1


def save_run(
    dataset_name,
    settings,
    distortions_used,
    distortion_levels,
    rf_results,
    lr_results,
    svm_results,
    fig_main,
    fig_analysis=None,
    analysis_df=None,
    # ── new reliability params ──────────────────────────────────────
    fig_reliability=None,
    horizons_df=None,
    reliability_threshold=None,
    reliability_metric=None,
):
    """
    Save a simulation run to disk.

    Parameters (original)
    ─────────────────────
    dataset_name       : str
    settings           : dict  {max_distortion_level, num_levels, test_size,
                                scale_data, random_seed, selected_models}
    distortions_used   : dict  {distortion_name: bool}
    distortion_levels  : list[float]
    rf_results         : list[dict]   per-level metrics for Random Forest
    lr_results         : list[dict]   per-level metrics for Logistic Regression
    svm_results        : list[dict]   per-level metrics for SVM
    fig_main           : matplotlib Figure   2x3 metrics chart
    fig_analysis       : matplotlib Figure   per-distortion bar chart (optional)
    analysis_df        : pd.DataFrame        per-distortion table (optional)

    Parameters (new — reliability)
    ──────────────────────────────
    fig_reliability        : matplotlib Figure   reliability windows chart (optional)
    horizons_df            : pd.DataFrame        reliability horizon table (optional)
    reliability_threshold  : float               threshold used (optional)
    reliability_metric     : str                 metric used (optional)

    Returns
    ───────
    run_id  : int
    run_dir : Path
    """
    run_id  = _get_next_run_id()
    run_dir = RESULTS_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save main chart ────────────────────────────────────────────────────────
    main_png = run_dir / "results.png"
    fig_main.savefig(main_png, dpi=120, bbox_inches="tight")

    # ── Save per-distortion chart ──────────────────────────────────────────────
    analysis_png = None
    if fig_analysis is not None:
        analysis_png = run_dir / "per_distortion.png"
        fig_analysis.savefig(analysis_png, dpi=120, bbox_inches="tight")

    # ── Save reliability chart ─────────────────────────────────────────────────
    reliability_png = None
    if fig_reliability is not None:
        reliability_png = run_dir / "reliability.png"
        fig_reliability.savefig(reliability_png, dpi=120, bbox_inches="tight")

    # ── Build reliability data for JSON ───────────────────────────────────────
    reliability_data = {}
    if horizons_df is not None and not horizons_df.empty:
        reliability_data = {
            "threshold":  reliability_threshold,
            "metric":     reliability_metric,
            "horizons":   horizons_df.to_dict(orient="records"),
        }

    # ── Build JSON payload ─────────────────────────────────────────────────────
    payload = {
        "run_id":          run_id,
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":         dataset_name,
        "settings":        settings,
        "distortions_used": distortions_used,
        "levels":          distortion_levels,
        "results": {
            "random_forest":        rf_results,
            "logistic_regression":  lr_results,
            "svm":                  svm_results,
        },
        "reliability":     reliability_data,
        "_png_path":          str(main_png),
        "_analysis_png_path": str(analysis_png)    if analysis_png    else None,
        "_reliability_png_path": str(reliability_png) if reliability_png else None,
    }

    # ── Also save per-distortion table ────────────────────────────────────────
    if analysis_df is not None:
        analysis_csv = run_dir / "per_distortion_analysis.csv"
        analysis_df.to_csv(analysis_csv, index=False)

    # ── Also save reliability horizon table ───────────────────────────────────
    if horizons_df is not None and not horizons_df.empty:
        horizons_csv = run_dir / "reliability_horizons.csv"
        horizons_df.to_csv(horizons_csv, index=False)

    # ── Write JSON ─────────────────────────────────────────────────────────────
    json_path = run_dir / f"run_{run_id:03d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return run_id, run_dir


def load_all_runs():
    """
    Load all saved runs from the results/ directory.
    Returns a list of run dicts, sorted by run_id ascending.
    """
    if not RESULTS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        json_files = list(run_dir.glob("run_*.json"))
        if not json_files:
            continue
        try:
            with open(json_files[0], "r") as f:
                data = json.load(f)
            runs.append(data)
        except Exception:
            continue

    return sorted(runs, key=lambda r: r.get("run_id", 0))
