# evaluation/metrics.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(labels, predictions):
    """
    Calculates accuracy, precision, recall, and F1-score for given labels and predictions.
    Uses weighted averaging to handle class imbalances.
    - labels: Ground truth labels.
    - predictions: Model predictions.
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    # Return metrics as a dictionary for easy access
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
