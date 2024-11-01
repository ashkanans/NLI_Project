# scripts/evaluate_models.py

import torch
import pandas as pd
from data.data_loader import load_dataset_splits, create_data_loader
from models.utils import initialize_baseline_model, initialize_enhanced_model, load_model
from evaluation.evaluate import evaluate_all_models
from config.config import CONFIG


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets and create DataLoaders
    _, _, test_data = load_dataset_splits()
    test_loader = create_data_loader(test_data, CONFIG["batch_size"])

    # Load adversarial test dataset if it's saved separately; otherwise, augment manually.
    # Assuming `load_adversarial_data()` is a function that provides the adversarial dataset
    from data.augmentation import augment_dataset
    adversarial_test_data = augment_dataset(test_data)
    adversarial_test_loader = create_data_loader(adversarial_test_data, CONFIG["batch_size"])

    # Initialize models
    baseline_model = initialize_baseline_model().to(device)
    enhanced_model = initialize_enhanced_model().to(device)

    # Load saved model checkpoints if available
    try:
        baseline_model = load_model(baseline_model, path=CONFIG["save_path"], filename="baseline_model_checkpoint.pt")
        enhanced_model = load_model(enhanced_model, path=CONFIG["save_path"], filename="enhanced_model_checkpoint.pt")
    except FileNotFoundError as e:
        print(f"Warning: Model checkpoint not found. Ensure models are trained before evaluation.\n{e}")

    # Evaluate all models and get results
    print("Evaluating all models on original and adversarial test sets...")
    results = evaluate_all_models(baseline_model, enhanced_model, test_loader, adversarial_test_loader, device)

    # Display results as a DataFrame
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df)

    # Optionally, save the results to a CSV file for later analysis
    results_df.to_csv(f"{CONFIG['save_path']}/evaluation_results.csv", index=False)
    print(f"Results saved to {CONFIG['save_path']}/evaluation_results.csv")


if __name__ == "__main__":
    main()
