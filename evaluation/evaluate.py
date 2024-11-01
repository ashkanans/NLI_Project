# evaluation/evaluate.py

import torch
from evaluation.metrics import calculate_metrics
from models.utils import load_model


def evaluate_model(model, dataloader, device, use_wsd_srl=False):
    """
    Evaluates a single model (baseline or enhanced) on a given DataLoader.
    - model: The model to evaluate.
    - dataloader: DataLoader with test or validation data.
    - device: Device to run the evaluation on (CPU/GPU).
    - use_wsd_srl: Boolean flag to indicate if WSD and SRL embeddings are used.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with or without WSD/SRL embeddings
            if use_wsd_srl:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    wsd_ids=batch["wsd_ids"],
                    srl_ids=batch["srl_ids"]
                )
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )

            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate and return evaluation metrics
    return calculate_metrics(all_labels, all_predictions)


def evaluate_ensemble(baseline_model, enhanced_model, dataloader, device):
    """
    Evaluates an ensemble of baseline and enhanced models by averaging logits.
    - baseline_model: The baseline model.
    - enhanced_model: The enhanced model with WSD and SRL embeddings.
    - dataloader: DataLoader with test or validation data.
    - device: Device to run the evaluation on (CPU/GPU).
    """
    baseline_model.eval()
    enhanced_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass for baseline model
            baseline_outputs = baseline_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            baseline_logits = baseline_outputs.logits

            # Forward pass for enhanced model with WSD and SRL embeddings
            enhanced_outputs = enhanced_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                wsd_ids=batch["wsd_ids"],
                srl_ids=batch["srl_ids"]
            )
            enhanced_logits = enhanced_outputs

            # Average logits from both models
            combined_logits = (baseline_logits + enhanced_logits) / 2
            predictions = torch.argmax(combined_logits, dim=-1)

            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate and return evaluation metrics
    return calculate_metrics(all_labels, all_predictions)


def evaluate_all_models(baseline_model, enhanced_model, dataloader, adversarial_dataloader, device):
    """
    Evaluates the baseline, enhanced, and ensemble models on both original and adversarial datasets.
    - Returns a dictionary with metrics for each model and dataset.
    """
    print("Evaluating Baseline Model on Original Test Set")
    baseline_metrics_original = evaluate_model(baseline_model, dataloader, device)

    print("Evaluating Baseline Model on Adversarial Test Set")
    baseline_metrics_adversarial = evaluate_model(baseline_model, adversarial_dataloader, device)

    print("Evaluating Enhanced Model on Original Test Set")
    enhanced_metrics_original = evaluate_model(enhanced_model, dataloader, device, use_wsd_srl=True)

    print("Evaluating Enhanced Model on Adversarial Test Set")
    enhanced_metrics_adversarial = evaluate_model(enhanced_model, adversarial_dataloader, device, use_wsd_srl=True)

    print("Evaluating Ensemble Model on Original Test Set")
    ensemble_metrics_original = evaluate_ensemble(baseline_model, enhanced_model, dataloader, device)

    print("Evaluating Ensemble Model on Adversarial Test Set")
    ensemble_metrics_adversarial = evaluate_ensemble(baseline_model, enhanced_model, adversarial_dataloader, device)

    # Compile results into a dictionary
    results = {
        "Model": ["Baseline", "Baseline", "Enhanced", "Enhanced", "Ensemble", "Ensemble"],
        "Dataset": ["Original", "Adversarial", "Original", "Adversarial", "Original", "Adversarial"],
        "Accuracy": [baseline_metrics_original["accuracy"], baseline_metrics_adversarial["accuracy"],
                     enhanced_metrics_original["accuracy"], enhanced_metrics_adversarial["accuracy"],
                     ensemble_metrics_original["accuracy"], ensemble_metrics_adversarial["accuracy"]],
        "Precision": [baseline_metrics_original["precision"], baseline_metrics_adversarial["precision"],
                      enhanced_metrics_original["precision"], enhanced_metrics_adversarial["precision"],
                      ensemble_metrics_original["precision"], ensemble_metrics_adversarial["precision"]],
        "Recall": [baseline_metrics_original["recall"], baseline_metrics_adversarial["recall"],
                   enhanced_metrics_original["recall"], enhanced_metrics_adversarial["recall"],
                   ensemble_metrics_original["recall"], ensemble_metrics_adversarial["recall"]],
        "F1-score": [baseline_metrics_original["f1"], baseline_metrics_adversarial["f1"],
                     enhanced_metrics_original["f1"], enhanced_metrics_adversarial["f1"],
                     ensemble_metrics_original["f1"], ensemble_metrics_adversarial["f1"]]
    }
    return results
