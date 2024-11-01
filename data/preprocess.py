# data/preprocess.py

import torch
from transformers import AutoTokenizer
from config.config import CONFIG

# Initialize the tokenizer using the specified model name
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


def get_wsd_srl_ids(annotation, vocab_size):
    """
    Helper function to convert WSD/SRL annotations into IDs based on vocab size.
    Uses a hash-based approach to map each unique annotation to a fixed vocabulary size.
    """
    ids = []
    for token in annotation:
        # Check if the annotation has a WSD or SRL ID; otherwise, use a placeholder
        if "bnSynsetId" in token and token["bnSynsetId"] != "O":
            # Hash synset ID and map to the fixed vocab range
            synset_id = hash(token["bnSynsetId"]) % vocab_size
        else:
            synset_id = 0  # Placeholder for missing annotation
        ids.append(synset_id)
    return ids


def preprocess_function(examples):
    """
    Preprocesses examples from the dataset for model training.
    Tokenizes premise-hypothesis pairs and extracts WSD/SRL annotations,
    padding them to a fixed length.
    """
    # Tokenize input text using premise-hypothesis format
    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"],
        return_tensors="pt"
    )

    # Map labels to integers
    label_map = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}
    inputs["labels"] = torch.tensor([label_map[label] for label in examples["label"]])

    # Extract WSD and SRL annotations for premise and hypothesis
    wsd_ids = []
    srl_ids = []

    for i, (premise, hypothesis) in enumerate(zip(examples["wsd"], examples["srl"])):
        # Get WSD IDs for premise and hypothesis, and concatenate them
        premise_wsd_ids = get_wsd_srl_ids(premise["premise"], CONFIG["wsd_vocab_size"])
        hypothesis_wsd_ids = get_wsd_srl_ids(hypothesis["hypothesis"], CONFIG["wsd_vocab_size"])
        wsd_ids.append(premise_wsd_ids + hypothesis_wsd_ids)

        # Get SRL IDs for premise and hypothesis, and concatenate them
        premise_srl_ids = get_wsd_srl_ids(premise["premise"], CONFIG["srl_vocab_size"])
        hypothesis_srl_ids = get_wsd_srl_ids(hypothesis["hypothesis"], CONFIG["srl_vocab_size"])
        srl_ids.append(premise_srl_ids + hypothesis_srl_ids)

    # Pad WSD and SRL IDs to match `max_length`, using 0 as the padding value
    inputs["wsd_ids"] = torch.tensor([
        wsd_id[:CONFIG["max_length"]] + [0] * (CONFIG["max_length"] - len(wsd_id))
        for wsd_id in wsd_ids
    ])
    inputs["srl_ids"] = torch.tensor([
        srl_id[:CONFIG["max_length"]] + [0] * (CONFIG["max_length"] - len(srl_id))
        for srl_id in srl_ids
    ])

    return inputs
