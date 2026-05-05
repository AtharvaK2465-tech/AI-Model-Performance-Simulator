from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RecommendRequest(BaseModel):
    results: list[dict]
    weights: dict = {"accuracy": 0.4, "f1": 0.3, "robustness": 0.3}

@router.post("/recommend")
def recommend(req: RecommendRequest):
    if not req.results:
        return {"error": "No results provided"}

    model_names = [k for k in req.results[0].keys() if k != "level"]
    scores = {}

    for model in model_names:
        accs = [r[model]["accuracy"] for r in req.results if model in r]
        f1s  = [r[model]["f1"]       for r in req.results if model in r]
        if not accs:
            continue

        avg_acc    = sum(accs) / len(accs)
        avg_f1     = sum(f1s)  / len(f1s)
        robustness = accs[-1] / accs[0] if accs[0] > 0 else 0

        w = req.weights
        composite = (
            w.get("accuracy",   0.4) * avg_acc +
            w.get("f1",         0.3) * avg_f1  +
            w.get("robustness", 0.3) * robustness
        )

        scores[model] = {
            "avg_accuracy":    round(avg_acc,    4),
            "avg_f1":          round(avg_f1,     4),
            "robustness":      round(robustness, 4),
            "composite_score": round(composite,  4)
        }

    if not scores:
        return {"error": "No model data found"}

    best = max(scores, key=lambda k: scores[k]["composite_score"])
    return {"scores": scores, "recommended": best}
