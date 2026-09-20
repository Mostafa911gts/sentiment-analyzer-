---
title: Sentiment Analyzer
emoji: 😊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Sentiment Analyzer

A small web app that classifies English text as **positive** or **negative** and shows the model's confidence.

## Problem
Quickly understanding the tone of customer reviews, feedback, or messages.

## How it works
- Model: `distilbert-base-uncased-finetuned-sst-2-english` (pretrained, from Hugging Face)
- Interface: Gradio
- Deployment: Hugging Face Spaces

## Run locally
```bash
pip install transformers torch gradio
python app.py
```

## Demo
Live demo: <https://huggingface.co/spaces/Mostafa911/sentiment-analyzer2>

## Limitations
- English only
- Binary output (positive / negative), so mixed or neutral text may be classified with lower confidence
