# ============================================================
# FASTAPI APPLICATION (SENTIMENT ANALYSIS INFERENCE SERVICE)
# ------------------------------------------------------------
# PURPOSE:
# Expose the champion sentiment model through a REST API.
#
# ARCHITECTURE:
#
# Client
#   ↓
# FastAPI
#   ↓
# src.inference.predict
#   ↓
# Champion model
#
# ENDPOINTS:
#
# GET /
#     Basic API information.
#
# GET /health
#     Health check endpoint.
#
# POST /predict
#     Sentiment prediction endpoint.
#
# DESIGN PRINCIPLES:
#
# - Keep ML logic outside the API layer.
# - Reuse training preprocessing exactly.
# - Load only the locked champion model.
# - Cloud-ready architecture.
#
# FUTURE DEPLOYMENTS:
#
# - Docker containers
# - AWS SageMaker endpoints
# - Azure ML Managed Online Endpoints
# - Vertex AI custom containers
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_text


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NLP Sentiment Analysis API",
    description="Production-ready sentiment analysis service.",
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):
    text: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "NLP Sentiment Analysis ML Engineering System",
        "status": "running",
        "model_selection": "champion-based",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    result = predict_text(request.text)

    return result