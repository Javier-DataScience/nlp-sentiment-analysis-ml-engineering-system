"""
Streamlit UI for NLP Sentiment Analysis System

Features:
- Champion model display (from artifacts/champion.json)
- Metrics dashboard (from artifacts/metrics.json)
- Text sentiment prediction via FastAPI or local inference
- Prediction history using session state (stateful UI)
"""

import streamlit as st
import requests
import json
from pathlib import Path

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🤖",
    layout="wide",
)

ARTIFACTS_PATH = Path("artifacts")

CHAMPION_PATH = ARTIFACTS_PATH / "champion.json"
METRICS_PATH = ARTIFACTS_PATH / "metrics.json"

# =========================
# SESSION STATE INIT
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOAD ARTIFACTS
# =========================


def load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


champion = load_json(CHAMPION_PATH)
metrics = load_json(METRICS_PATH)

# =========================
# API CONFIG
# =========================

API_URL = "http://localhost:8000/predict"


def predict_sentiment(text: str):
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=10,
        )
        return response.json()
    except Exception as e:
        return {
            "prediction": "error",
            "confidence": 0.0,
            "error": str(e),
        }


# =========================
# UI LAYOUT
# =========================

st.title("🤖 NLP Sentiment Analysis System")

# =========================
# SIDEBAR - CHAMPION + METRICS
# =========================

with st.sidebar:
    st.header("🏆 Champion Model")

    if champion:
        st.write(f"**Model:** {champion.get('model')}")
        st.write(f"**Run ID:** {champion.get('run_id')}")
        st.write(f"**Metric:** {champion.get('metric')}")
        st.write(f"**Score:** {champion.get('score')}")
    else:
        st.write("No champion found")

    st.divider()

    st.header("📊 Metrics")

    if metrics:
        for model_name, model_metrics in metrics.items():
            st.subheader(model_name)
            for k, v in model_metrics.items():
                st.write(f"{k}: {v}")
    else:
        st.write("No metrics found")

# =========================
# MAIN INPUT
# =========================

st.subheader("📝 Enter text for sentiment analysis")

user_input = st.text_area("Text input", height=120)

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_sentiment(user_input)

        prediction = result.get("prediction", "unknown")
        confidence = result.get("confidence", 0.0)

        # =========================
        # STORE IN HISTORY (STATE)
        # =========================

        st.session_state.history.append(
            {
                "text": user_input,
                "prediction": prediction,
                "confidence": confidence,
            }
        )

        # =========================
        # DISPLAY RESULT
        # =========================

        st.success("Prediction completed")

        st.markdown(f"### Prediction: `{prediction.upper()}`")
        st.markdown(f"### Confidence: `{confidence:.2f}`")

# =========================
# HISTORY SECTION
# =========================

st.divider()

st.subheader("📜 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions yet.")
else:
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"### {i}. {item['prediction'].upper()} ({item['confidence']:.2f})")
        st.write(item["text"])
        st.divider()
