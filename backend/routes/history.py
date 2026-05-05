from fastapi import APIRouter
import json, os

router = APIRouter()
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

@router.get("/history")
def history():
    runs = []
    if not os.path.exists(RESULTS_DIR):
        return {"runs": []}
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                runs.append(json.load(f))
    return {"runs": runs}
