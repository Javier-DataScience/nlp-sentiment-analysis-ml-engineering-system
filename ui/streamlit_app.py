"""
Streamlit UI for NLP Sentiment Analysis System (Refactored)

ARCHITECTURE:
Streamlit → API Client → FastAPI → Model

No direct inference logic in UI.
"""

import streamlit as st
import json
from pathlib import Path

from src.client.api_client import SentimentAPIClient

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
# API CLIENT
# =========================

client = SentimentAPIClient()

# =========================
# SESSION STATE
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
# UI
# =========================

st.title("🤖 NLP Sentiment Analysis System")

# =========================
# SIDEBAR
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

        # CALL API CLIENT
        result = client.predict(user_input)

        prediction = result.get("prediction", "unknown")
        confidence = result.get("confidence", 0.0)

        # STORE HISTORY
        st.session_state.history.append(
            {
                "text": user_input,
                "prediction": prediction,
                "confidence": confidence,
            }
        )

        # OUTPUT
        if prediction == "error":
            st.error("FastAPI request failed")
            st.text(result.get("error"))
        else:
            st.success("Prediction completed")
            st.markdown(f"### Prediction: `{prediction.upper()}`")
            st.markdown(f"### Confidence: `{confidence:.2f}`")

# =========================
# HISTORY
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
