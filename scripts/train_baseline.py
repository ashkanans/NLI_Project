# scripts/train_baseline.py

import torch
from data.data_loader import load_dataset_splits, create_data_loader
from models.utils import initialize_baseline_model, save_model
from training.train import train_model
from config.config import CONFIG


def main():
    # Load datasets and create DataLoaders
    train_data, val_data, _ = load_dataset_splits()
    train_loader = create_data_loader(train_data, CONFIG["batch_size"], shuffle=True)
    val_loader = create_data_loader(val_data, CONFIG["batch_size"])

    # Initialize baseline model and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_baseline_model().to(device)

    # Train the model and save after training
    print("Starting training for baseline model...")
    trained_model = train_model(model, train_loader, val_loader, device, model_type="baseline")

    # Save the trained model
    save_model(trained_model, path=CONFIG["save_path"], filename="baseline_model_checkpoint.pt")
    print("Baseline model training complete and saved.")


if __name__ == "__main__":
    main()
