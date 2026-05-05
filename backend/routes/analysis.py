from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class AnalysisRequest(BaseModel):
    results: list[dict]

@router.post("/analysis")
def analysis(req: AnalysisRequest):
    if not req.results:
        return {"error": "No results provided"}

    baseline = req.results[0]
    model_names = [k for k in baseline.keys() if k != "level"]

    # Degradation — % drop from baseline for each metric
    degradation = []
    for row in req.results:
        d = {"level": row["level"]}
        for model in model_names:
            if model in row and model in baseline:
                for metric in ["accuracy", "f1", "roc_auc"]:
                    base_val = baseline[model][metric]
                    curr_val = row[model][metric]
                    drop = round((base_val - curr_val) / base_val * 100, 2) if base_val > 0 else 0
                    d[f"{model}_{metric}_drop"] = drop
        degradation.append(d)

    # Reliability — stability across distortion levels
    reliability = {}
    for model in model_names:
        accs = [r[model]["accuracy"] for r in req.results if model in r]
        if accs:
            mean = sum(accs) / len(accs)
            variance = sum((a - mean) ** 2 for a in accs) / len(accs)
            std = round(variance ** 0.5, 4)
            reliability[model] = {
                "mean_accuracy":   round(mean, 4),
                "std_dev":         std,
                "stability_score": round(max(0, 1 - std), 4)
            }

    return {
        "per_distortion": req.results,
        "degradation":    degradation,
        "reliability":    reliability
    }
