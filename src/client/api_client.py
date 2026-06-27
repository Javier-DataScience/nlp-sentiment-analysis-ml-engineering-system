"""
Central API Client for ML Inference Service

PURPOSE:
This module centralizes all HTTP communication with FastAPI.
It is shared across:
- Streamlit UI
- Gradio UI

BENEFITS:
- No duplicated request logic
- Single point of change for API communication
- Cleaner architecture (production-style separation)
"""

import requests


class SentimentAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.predict_url = f"{self.base_url}/predict"

    def predict(self, text: str) -> dict:
        """
        Sends text to FastAPI and returns prediction result.
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
                "error": str(e),
            }