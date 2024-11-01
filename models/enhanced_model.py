# models/enhanced_model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from config.config import CONFIG


class EnhancedNLIModel(nn.Module):
    """
    Enhanced model that incorporates WSD and SRL embeddings with a pre-trained
    transformer model for Natural Language Inference.
    """

    def __init__(self, model_name, wsd_vocab_size, srl_vocab_size, num_labels=3):
        super(EnhancedNLIModel, self).__init__()

        # Load the base transformer model (e.g., DeBERTa)
        self.base_model = AutoModel.from_pretrained(model_name)

        # Define the embedding layers for WSD and SRL information
        hidden_size = self.base_model.config.hidden_size
        self.wsd_embedding = nn.Embedding(wsd_vocab_size, hidden_size)
        self.srl_embedding = nn.Embedding(srl_vocab_size, hidden_size)

        # A linear layer to project concatenated embeddings back to the hidden size
        self.fc = nn.Linear(hidden_size * 3, hidden_size)

        # Final classification layer for entailment, contradiction, and neutral
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, wsd_ids, srl_ids):
        # Forward pass through the base transformer model
        base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        base_embeddings = base_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Forward pass through the WSD and SRL embedding layers
        wsd_embeddings = self.wsd_embedding(wsd_ids)  # Shape: (batch_size, seq_len, hidden_size)
        srl_embeddings = self.srl_embedding(srl_ids)  # Shape: (batch_size, seq_len, hidden_size)

        # Concatenate the embeddings along the hidden dimension
        combined_embeddings = torch.cat((base_embeddings, wsd_embeddings, srl_embeddings), dim=-1)

        # Project the concatenated embeddings back to the hidden size
        combined_embeddings = self.fc(combined_embeddings)  # Shape: (batch_size, seq_len, hidden_size)

        # Use the [CLS] token (first token in sequence) for classification
        cls_embedding = combined_embeddings[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Final classification logits
        logits = self.classifier(cls_embedding)  # Shape: (batch_size, num_labels)

        return logits


def create_enhanced_model():
    """
    Factory function to create an instance of the EnhancedNLIModel using
    configurations defined in config/config.py.
    """
    model = EnhancedNLIModel(
        model_name=CONFIG["model_name"],
        wsd_vocab_size=CONFIG["wsd_vocab_size"],
        srl_vocab_size=CONFIG["srl_vocab_size"],
        num_labels=3
    )
    return model
