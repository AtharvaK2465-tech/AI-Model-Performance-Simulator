import json
import os
from datetime import datetime

RESULTS_DIR = "results"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _next_run_id():
    _ensure_dir(RESULTS_DIR)
    existing = [
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d.startswith("run_")
    ]
    return len(existing) + 1


def save_run(ds_name, settings, distortions_used, levels, rf_results, lr_results, svm_results, fig):
    """Save a simulation run to results/run_NNN/ folder."""
    run_id  = _next_run_id()
    run_dir = os.path.join(RESULTS_DIR, f"run_{run_id:03d}")
    _ensure_dir(run_dir)

    # Save JSON
    payload = {
        "run_id":           run_id,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":          ds_name,
        "settings":         settings,
        "distortions_used": distortions_used,
        "levels":           levels,
        "results": {
            "random_forest":       rf_results,
            "logistic_regression": lr_results,
            "svm":                 svm_results,
        }
    }
    json_path = os.path.join(run_dir, f"run_{run_id:03d}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save chart for this run
    png_path = os.path.join(run_dir, "results.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")

    # Also overwrite root results.png for README
    fig.savefig("results.png", dpi=150, bbox_inches="tight")

    return run_id, run_dir


def load_all_runs():
    """Load all saved runs from results/ folder."""
    _ensure_dir(RESULTS_DIR)
    runs = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(run_dir) or not d.startswith("run_"):
            continue
        json_file = os.path.join(run_dir, f"{d}.json")
        png_file  = os.path.join(run_dir, "results.png")
        if not os.path.exists(json_file):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            data["_png_path"] = png_file if os.path.exists(png_file) else None
            runs.append(data)
        except Exception:
            pass
    return runs
