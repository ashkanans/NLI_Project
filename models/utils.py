# models/utils.py

import os
import torch
from transformers import AutoModelForSequenceClassification
from config.config import CONFIG


def save_model(model, path, filename="model_checkpoint.pt"):
    """
    Saves the model's state dictionary to the specified path.
    Creates the directory if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def load_model(model, path, filename="model_checkpoint.pt"):
    """
    Loads the model's state dictionary from the specified path.
    Returns the model with the loaded state.
    """
    load_path = os.path.join(path, filename)
    if not os.path.isfile(load_path):
        raise FileNotFoundError(f"No model file found at {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
    print(f"Model loaded from {load_path}")
    return model


def initialize_baseline_model():
    """
    Initializes the baseline transformer model for sequence classification,
    with configurations set in config/config.py.
    """
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)
    return model


def initialize_enhanced_model():
    """
    Initializes the enhanced NLI model with WSD and SRL embeddings.
    Uses configurations set in config/config.py.
    """
    from models.enhanced_model import EnhancedNLIModel
    model = EnhancedNLIModel(
        model_name=CONFIG["model_name"],
        wsd_vocab_size=CONFIG["wsd_vocab_size"],
        srl_vocab_size=CONFIG["srl_vocab_size"],
        num_labels=3
    )
    return model
