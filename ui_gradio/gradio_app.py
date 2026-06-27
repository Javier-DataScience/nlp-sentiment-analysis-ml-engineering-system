"""
Gradio UI for NLP Sentiment Analysis System (Refactored)

ARCHITECTURE:
Gradio → API Client → FastAPI → Model

FIXES:
- Uses ONLY champion model metrics (no structural change)
- Removes direct inference dependency (predict_text)
- Keeps original UI layout intact
"""

import gradio as gr
import json
from pathlib import Path

from src.client.api_client import SentimentAPIClient

# =========================
# PATHS
# =========================

ARTIFACTS_PATH = Path("artifacts")

CHAMPION_PATH = ARTIFACTS_PATH / "champion.json"
METRICS_PATH = ARTIFACTS_PATH / "metrics.json"

# =========================
# API CLIENT
# =========================

client = SentimentAPIClient()

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
# INFERENCE FUNCTION (UPDATED)
# =========================

def predict(text, history):
    if not text or text.strip() == "":
        return "Please enter text.", history

    # CALL FASTAPI VIA CLIENT (NEW)
    result = client.predict(text)

    prediction = result.get("prediction", "unknown")
    confidence = result.get("confidence", 0.0)

    history.append({
        "text": text,
        "prediction": prediction,
        "confidence": confidence
    })

    return f"{prediction.upper()} ({confidence:.2f})", history

# =========================
# HISTORY FORMATTER
# =========================

def format_history(history):
    if not history:
        return "No predictions yet."

    output = ""
    for i, item in enumerate(reversed(history), 1):
        output += (
            f"{i}. {item['prediction'].upper()} ({item['confidence']:.2f})\n"
            f"   {item['text']}\n\n"
        )
    return output

# =========================
# GRADIO APP (UNCHANGED UI)
# =========================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    history_state = gr.State([])

    # =========================
    # TITLE
    # =========================
    gr.Markdown("# 🤖 Sentiment Analysis System")

    # =========================
    # MAIN LAYOUT
    # =========================

    with gr.Row():

        # =========================
        # LEFT PANEL (CHAMPION ONLY)
        # =========================
        with gr.Column(scale=1):

            gr.Markdown("## 🏆 Champion Model")

            if champion:
                model_name = champion.get("model")
                gr.Markdown(f"### {model_name}")

                model_metrics = metrics.get(model_name, {})

                if model_metrics:
                    with gr.Group():
                        for k, v in model_metrics.items():
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown(f"**{k}**")
                                with gr.Column(scale=1):
                                    gr.Markdown(f"{v}")
                else:
                    gr.Markdown("No metrics found for champion model")

                gr.Markdown("---")
                gr.Markdown(f"**Metric Used:** {champion.get('metric')}")
                gr.Markdown(f"**Score:** {champion.get('score')}")
                gr.Markdown(f"**Run ID:** {champion.get('run_id')}")

            else:
                gr.Markdown("No champion model found")

        # =========================
        # RIGHT PANEL (MAIN UI)
        # =========================
        with gr.Column(scale=2):

            gr.Markdown("## 📝 Enter Text")

            text_input = gr.Textbox(
                placeholder="Type a movie review...",
                lines=5,
                label=None
            )

            btn = gr.Button("Predict", variant="primary")

            output = gr.Textbox(
                label="Prediction Result",
                interactive=False
            )

            gr.Markdown("## 📜 Prediction History")

            history_box = gr.Textbox(
                lines=10,
                interactive=False
            )

            # =========================
            # EVENTS
            # =========================

            btn.click(
                fn=predict,
                inputs=[text_input, history_state],
                outputs=[output, history_state]
            ).then(
                fn=format_history,
                inputs=[history_state],
                outputs=[history_box]
            )

# =========================
# LAUNCH
# =========================

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )