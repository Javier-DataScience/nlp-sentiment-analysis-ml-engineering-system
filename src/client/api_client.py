# ============================================================
# CENTRAL API CLIENT FOR ML INFERENCE SERVICE
# ------------------------------------------------------------
# PURPOSE:
# Centralize all HTTP communication with the FastAPI service.
#
# SHARED BY:
# - Streamlit UI
# - Gradio UI
#
# ARCHITECTURE:
#
# Streamlit / Gradio
#          ↓
# SentimentAPIClient
#          ↓
# FastAPI
#          ↓
# Champion Model
#
# BENEFITS:
#
# - No duplicated request logic
# - Single source of truth for API communication
# - Environment-aware configuration
# - Docker-ready
# - Cloud-ready (AWS, Azure, GCP)
#
# ENVIRONMENT VARIABLE:
#
# API_URL
#
# Examples:
#
# Local development:
#     API_URL=http://127.0.0.1:8000
#
# Docker Compose:
#     API_URL=http://api:8000
#
# AWS:
#     API_URL=https://my-api.amazonaws.com
#
# Azure:
#     API_URL=https://my-api.azurewebsites.net
#
# GCP:
#     API_URL=https://my-api.a.run.app
#
# If API_URL is not defined, localhost is used by default.
# ============================================================

import os

import requests


class SentimentAPIClient:
    """
    Client responsible for communicating with the FastAPI
    inference service.
    """

    def __init__(self, base_url: str | None = None):

        self.base_url = base_url or os.getenv(
            "API_URL",
            "http://127.0.0.1:8000",
        )

        self.predict_url = f"{self.base_url}/predict"

    def predict(self, text: str) -> dict:
        """
        Send text to FastAPI and return prediction results.

        Parameters
        ----------
        text : str
            Input text for sentiment analysis.

        Returns
        -------
        dict
            Dictionary containing:
            - prediction
            - confidence
            - optional error message
        """

        try:

            response = requests.post(
                self.predict_url,
                json={"text": text},
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            return {
                "prediction": data.get("prediction", "unknown"),
                "confidence": data.get("confidence", 0.0),
            }

        except requests.exceptions.RequestException as e:

            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": f"FastAPI request failed\n\n{e}",
            }
