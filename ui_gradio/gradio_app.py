"""
Gradio UI for NLP Sentiment Analysis System

PURPOSE:
Provide a web interface for interacting with the sentiment
analysis service through FastAPI.

ARCHITECTURE:

User
  ↓
Gradio UI
  ↓
SentimentAPIClient
  ↓
FastAPI
  ↓
src.inference.predict
  ↓
Champion Model

FEATURES:
- Champion model information
- Champion metrics only
- Prediction history
- Shared API client (no duplicated HTTP logic)
- Docker-compatible launch configuration
"""

import json
from pathlib import Path

import gradio as gr

from src.client.api_client import SentimentAPIClient

# ==========================================================
# PATHS
# ==========================================================

ARTIFACTS_PATH = Path("artifacts")

CHAMPION_PATH = ARTIFACTS_PATH / "champion.json"
METRICS_PATH = ARTIFACTS_PATH / "metrics.json"


# ==========================================================
# LOAD JSON UTILITIES
# ==========================================================


def load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    return {}


champion = load_json(CHAMPION_PATH)
metrics = load_json(METRICS_PATH)


# ==========================================================
# API CLIENT
# ==========================================================

client = SentimentAPIClient()


# ==========================================================
# INFERENCE FUNCTION
# ==========================================================


def predict(text, history):

    if not text or text.strip() == "":
        return "Please enter text.", history

    result = client.predict(text)

    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)

    history.append(
        {
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
        }
    )

    return f"{prediction.upper()} ({confidence:.2f})", history


# ==========================================================
# HISTORY FORMATTER
# ==========================================================


def format_history(history):

    if not history:
        return "No predictions yet."

    output = ""

    for i, item in enumerate(reversed(history), 1):

        output += (
            f"{i}. {item['prediction'].upper()} "
            f"({item['confidence']:.2f})\n"
            f"   {item['text']}\n\n"
        )

    return output


# ==========================================================
# GRADIO APPLICATION
# ==========================================================

with gr.Blocks() as app:

    history_state = gr.State([])

    # ======================================================
    # TITLE
    # ======================================================

    gr.Markdown("# 🤖 Sentiment Analysis System")

    # ======================================================
    # MAIN LAYOUT
    # ======================================================

    with gr.Row():

        # ==================================================
        # LEFT PANEL
        # ==================================================

        with gr.Column(scale=1):

            gr.Markdown("## 🏆 Champion Model")

            if champion:

                model_name = champion.get("model")

                gr.Markdown(f"### {model_name}")

                model_metrics = metrics.get(model_name, {})

                if model_metrics:

                    with gr.Group():

                        for metric_name, metric_value in model_metrics.items():

                            with gr.Row():

                                with gr.Column(scale=1):
                                    gr.Markdown(f"**{metric_name}**")

                                with gr.Column(scale=1):
                                    gr.Markdown(str(metric_value))

                else:
                    gr.Markdown("No metrics found.")

                gr.Markdown("---")

                gr.Markdown(f"**Metric Used:** {champion.get('metric')}")

                gr.Markdown(f"**Score:** {champion.get('score')}")

                gr.Markdown(f"**Run ID:** {champion.get('run_id')}")

            else:
                gr.Markdown("No champion model found.")

        # ==================================================
        # RIGHT PANEL
        # ==================================================

        with gr.Column(scale=2):

            gr.Markdown("## 📝 Enter Text")

            text_input = gr.Textbox(
                placeholder="Type a movie review...",
                lines=5,
                label=None,
            )

            predict_button = gr.Button(
                "Predict",
                variant="primary",
            )

            prediction_output = gr.Textbox(
                label="Prediction Result",
                interactive=False,
            )

            gr.Markdown("## 📜 Prediction History")

            history_box = gr.Textbox(
                lines=10,
                interactive=False,
            )

            # ==============================================
            # EVENTS
            # ==============================================

            predict_button.click(
                fn=predict,
                inputs=[text_input, history_state],
                outputs=[prediction_output, history_state],
            ).then(
                fn=format_history,
                inputs=[history_state],
                outputs=[history_box],
            )


# ==========================================================
# APPLICATION ENTRYPOINT
# ==========================================================

if __name__ == "__main__":

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
