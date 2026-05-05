from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import simulate, analysis, recommend, history

app = FastAPI(title="AI Model Performance Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulate.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(history.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "AIMPS backend running"}
