# NOTE: "spaces" must be imported first (required by Hugging Face ZeroGPU)
try:
    import spaces
except ImportError:  # running locally or on Colab
    class spaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda fn: fn

import gradio as gr
from transformers import pipeline

# Pretrained model: no training or dataset needed
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


@spaces.GPU
def analyze(text):
    if not text or not text.strip():
        return {}, "Please type a sentence first."

    result = classifier(text, truncation=True, max_length=512)[0]
    label = result["label"].title()  # Positive / Negative
    score = float(result["score"])
    other = "Negative" if label == "Positive" else "Positive"

    scores = {label: score, other: 1 - score}
    emoji = "😊" if label == "Positive" else "😞"
    summary = f"{emoji} **{label}** ({score:.1%} confidence)"
    return scores, summary


EXAMPLES = [
    "I absolutely love this product, it works perfectly!",
    "The service was slow and the staff were rude.",
    "The movie was okay, but the ending felt rushed.",
]

with gr.Blocks(title="Sentiment Analyzer") as demo:
    gr.Markdown(
        "# Sentiment Analyzer\n"
        "Type an English sentence and see whether it sounds positive or negative."
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Your text",
                placeholder="e.g. I really enjoyed this course!",
                lines=4,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            chart = gr.Label(label="Confidence", num_top_classes=2)

    clear_btn.add([text, summary, chart])
    gr.Examples(examples=EXAMPLES, inputs=text)

    submit_btn.click(analyze, inputs=text, outputs=[chart, summary])
    text.submit(analyze, inputs=text, outputs=[chart, summary])

    gr.Markdown(
        "<small>Model: DistilBERT fine-tuned on SST-2 (Hugging Face). "
        "Built with Gradio.</small>"
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), ssr_mode=False)
