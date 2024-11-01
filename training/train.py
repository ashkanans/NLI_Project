# training/train.py
import torch
from tqdm import tqdm
from transformers import get_scheduler, AdamW
from config.config import CONFIG
from models.utils import save_model


def train_model(model, train_loader, validation_loader, device, model_type="baseline"):
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    num_training_steps = CONFIG["epochs"] * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})

        save_model(model, f"{CONFIG['save_path']}/{model_type}_epoch_{epoch + 1}.pt")
    return model
