# models/baseline_model.py
import torch
from transformers import AutoModelForSequenceClassification
from config.config import CONFIG


def create_baseline_model():
    return AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=3
    )
