from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json, asyncio
from core.model import train_models, ALL_MODELS
from core.distortions import apply_distortion
from core.evaluation import evaluate
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

router = APIRouter()

class SimulateRequest(BaseModel):
    dataset: str = "iris"
    models: list[str] = ["Random Forest", "Logistic Regression"]
    distortion_type: str = "noise"
    levels: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4]

@router.post("/simulate")
async def simulate(req: SimulateRequest):
    async def event_stream():
        # Load dataset
        if req.dataset == "breast_cancer":
            data = load_breast_cancer()
        else:
            data = load_iris()

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Filter to valid models only
        selected = [m for m in req.models if m in ALL_MODELS]
        if not selected:
            yield f"data: {json.dumps({'error': 'No valid models selected'})}\n\n"
            return

        trained = train_models(X_train, y_train, selected_models=selected)
        all_results = []

        for i, level in enumerate(req.levels):
            # apply_distortion needs both X and y, returns both
            distortion_kwargs = {}
            if req.distortion_type == "noise":
                distortion_kwargs["noise_level"] = level
            elif req.distortion_type == "drift":
                distortion_kwargs["drift_level"] = level
            elif req.distortion_type == "missing":
                distortion_kwargs["missing_level"] = level
            elif req.distortion_type == "imbalance":
                distortion_kwargs["imbalance_level"] = level
            elif req.distortion_type == "outlier":
                distortion_kwargs["outlier_level"] = level
            else:
                distortion_kwargs["noise_level"] = level  # default

            X_dist, y_dist = apply_distortion(X_test, y_test, **distortion_kwargs)

            row = {"level": level}

            for model_name, model_obj in trained.items():
                metrics = evaluate(model_obj, X_dist, y_dist)
                row[model_name] = {
                    "accuracy":   round(metrics["accuracy"],   4),
                    "precision":  round(metrics["precision"],  4),
                    "recall":     round(metrics["recall"],     4),
                    "f1":         round(metrics["f1"],         4),
                    "roc_auc":    round(metrics["roc_auc"],    4),
                    "confidence": round(metrics["confidence"], 4),
                }

            all_results.append(row)

            progress = {
                "step":  i + 1,
                "total": len(req.levels),
                "level": level,
                "done":  False
            }
            yield f"data: {json.dumps(progress)}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'done': True, 'results': all_results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
