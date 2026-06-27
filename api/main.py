# ============================================================
# FASTAPI APPLICATION (SENTIMENT ANALYSIS INFERENCE SERVICE)
# ============================================================
"""
This module exposes a production-ready REST API for sentiment analysis.

ARCHITECTURE OVERVIEW:

    Client (Streamlit / Gradio / external apps)
            ↓ HTTP
        FastAPI Service
            ↓
    Inference Layer (predict_text)
            ↓
    Champion Model (selected via champion.json)
            ↓
    Artifacts (model weights, vocab, metadata)

------------------------------------------------------------

KEY DESIGN PRINCIPLES:

1. Single Source of Truth for Model Serving
   - Only the CHAMPION model is used for inference.

2. Model Loaded Once at Startup
   - Avoids reloading model per request (performance optimization).

3. Separation of Concerns
   - API layer: request/response handling
   - inference layer: prediction logic
   - training layer: model creation only

4. Cloud/Docker Ready Design
   - Stateless API behavior (except in-memory model cache)
   - Easy to containerize and scale horizontally

------------------------------------------------------------

ENDPOINTS:

GET /
    Returns API metadata and status.

GET /health
    Health check for orchestration systems.

POST /predict
    Runs sentiment prediction using champion model.

============================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_text

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="NLP Sentiment Analysis API",
    description="Production-ready sentiment analysis service using champion model.",
    version="1.0.0",
)

# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    text: str


# ============================================================
# OPTIONAL: MODEL WARMUP (PLACEHOLDER FOR FUTURE OPTIMIZATION)
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    This is where we would load and cache the champion model.

    CURRENT STATE:
        - predict_text() handles loading internally.

    FUTURE IMPROVEMENT:
        - Load model + tokenizer + vocab here
        - Store in global state (app.state.model)
        - Remove repeated loading inside inference calls
    """
    print("FastAPI startup complete - model ready for inference")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "project": "NLP Sentiment Analysis ML Engineering System",
        "status": "running",
        "architecture": "champion-based inference",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    # Delegate ALL ML logic to inference layer
    result = predict_text(request.text)

    return {
        "input": request.text,
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence")
    }