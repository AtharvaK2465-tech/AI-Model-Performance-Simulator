"""
run_logger.py
Saves every simulation run to results/run_NNN/ with:
  - run_NNN.json               (all metadata + results + reliability data)
  - results.png                (main 2x3 metrics chart)
  - per_distortion.png         (per-distortion bar chart)
  - reliability.png            (reliability horizon chart)
  - confusion_matrices.png     (clean vs max distortion confusion matrices)
  - degradation_analysis.png   (degradation rate + robustness ranking)
  - recommendation_dashboard.png  (model recommendation engine chart)  ← NEW
  - per_distortion_analysis.csv
  - reliability_horizons.csv
  - degradation_summary.csv
  - model_recommendation.csv                                           ← NEW
"""

import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

RESULTS_DIR = Path("results")


def _get_next_run_id():
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ])
    if not existing:
        return 1
    last = existing[-1].name
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
    fig_reliability=None,
    horizons_df=None,
    reliability_threshold=None,
    reliability_metric=None,
    fig_confusion=None,
    fig_degradation=None,
    summary_df=None,
    fig_recommendation=None,   # ← NEW
    rec_df=None,               # ← NEW
):
    run_id  = _get_next_run_id()
    run_dir = RESULTS_DIR / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save charts ────────────────────────────────────────────────────────────
    main_png = run_dir / "results.png"
    fig_main.savefig(main_png, dpi=120, bbox_inches="tight")

    analysis_png = None
    if fig_analysis is not None:
        analysis_png = run_dir / "per_distortion.png"
        fig_analysis.savefig(analysis_png, dpi=120, bbox_inches="tight")

    reliability_png = None
    if fig_reliability is not None:
        reliability_png = run_dir / "reliability.png"
        fig_reliability.savefig(reliability_png, dpi=120, bbox_inches="tight")

    confusion_png = None
    if fig_confusion is not None:
        confusion_png = run_dir / "confusion_matrices.png"
        fig_confusion.savefig(confusion_png, dpi=120, bbox_inches="tight")

    degradation_png = None
    if fig_degradation is not None:
        degradation_png = run_dir / "degradation_analysis.png"
        fig_degradation.savefig(degradation_png, dpi=120, bbox_inches="tight")

    recommendation_png = None                                          # ← NEW
    if fig_recommendation is not None:                                 # ← NEW
        recommendation_png = run_dir / "recommendation_dashboard.png" # ← NEW
        fig_recommendation.savefig(                                    # ← NEW
            recommendation_png, dpi=120, bbox_inches="tight"          # ← NEW
        )                                                              # ← NEW

    # ── Build reliability data for JSON ───────────────────────────────────────
    reliability_data = {}
    if horizons_df is not None and not horizons_df.empty:
        reliability_data = {
            "threshold": reliability_threshold,
            "metric":    reliability_metric,
            "horizons":  horizons_df.to_dict(orient="records"),
        }

    # ── Build degradation data for JSON ───────────────────────────────────────
    degradation_data = {}
    if summary_df is not None and not summary_df.empty:
        degradation_data = summary_df.to_dict(orient="records")

    # ── Build recommendation data for JSON ────────────────────────────────────  ← NEW
    recommendation_data = {}                                           # ← NEW
    if rec_df is not None and not rec_df.empty:                        # ← NEW
        recommendation_data = rec_df.to_dict(orient="records")        # ← NEW

    # ── Build JSON payload ─────────────────────────────────────────────────────
    payload = {
        "run_id":                      run_id,
        "timestamp":                   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":                     dataset_name,
        "settings":                    settings,
        "distortions_used":            distortions_used,
        "levels":                      distortion_levels,
        "results": {
            "random_forest":           rf_results,
            "logistic_regression":     lr_results,
            "svm":                     svm_results,
        },
        "reliability":                 reliability_data,
        "degradation":                 degradation_data,
        "recommendation":              recommendation_data,            # ← NEW
        "_png_path":                   str(main_png),
        "_analysis_png_path":          str(analysis_png)         if analysis_png         else None,
        "_reliability_png_path":       str(reliability_png)      if reliability_png      else None,
        "_confusion_png_path":         str(confusion_png)        if confusion_png        else None,
        "_degradation_png_path":       str(degradation_png)      if degradation_png      else None,
        "_recommendation_png_path":    str(recommendation_png)   if recommendation_png   else None,  # ← NEW
    }

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    if analysis_df is not None:
        analysis_df.to_csv(run_dir / "per_distortion_analysis.csv", index=False)

    if horizons_df is not None and not horizons_df.empty:
        horizons_df.to_csv(run_dir / "reliability_horizons.csv", index=False)

    if summary_df is not None and not summary_df.empty:
        summary_df.to_csv(run_dir / "degradation_summary.csv", index=False)

    if rec_df is not None and not rec_df.empty:                        # ← NEW
        rec_df.to_csv(                                                 # ← NEW
            run_dir / "model_recommendation.csv", index=False          # ← NEW
        )                                                              # ← NEW

    # ── Write JSON ─────────────────────────────────────────────────────────────
    json_path = run_dir / f"run_{run_id:03d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return run_id, run_dir


def load_all_runs():
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
