---

# NLI Project with Enhanced Transformer Model

This project implements a Natural Language Inference (NLI) system using transformer-based models, including a baseline model and an enhanced model with Word Sense Disambiguation (WSD) and Semantic Role Labeling (SRL) embeddings. The project aims to improve NLI robustness by using data augmentation to create adversarial examples and leveraging ensemble evaluations.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training the Baseline Model](#training-the-baseline-model)
  - [Training the Enhanced Model](#training-the-enhanced-model)
  - [Evaluating Models](#evaluating-models)
- [Configuration](#configuration)
- [Detailed Module Explanations](#detailed-module-explanations)

---

## Overview

The project is structured to facilitate training and evaluating NLI models, specifically:

- **Baseline Model**: A transformer-based model for standard NLI classification.
- **Enhanced Model**: An advanced NLI model that incorporates WSD and SRL embeddings for improved semantic
  understanding.
- **Data Augmentation**: Methods to generate adversarial samples by substituting synonyms and antonyms, creating more
  challenging test cases.
- **Evaluation**: Evaluates baseline, enhanced, and ensemble models on original and adversarial test sets.

The project structure modularizes data loading, augmentation, model training, and evaluation for easy modification and
expansion.

---

## Project Structure

```
NLI_Project/
├── config/
│   └── config.py                 # Configuration settings for paths, hyperparameters, and model names
├── data/
│   ├── data_loader.py            # Functions for loading and processing datasets
│   ├── augmentation.py           # Functions for data augmentation with synonyms and antonyms
│   └── preprocess.py             # Tokenization, WSD, and SRL embedding processing
├── models/
│   ├── baseline_model.py         # Definition of the baseline transformer model
│   ├── enhanced_model.py         # Enhanced model with WSD and SRL embeddings
│   └── utils.py                  # Model saving, loading, and initialization utilities
├── training/
│   └── train.py                  # Training functions for baseline and enhanced models
├── evaluation/
│   ├── evaluate.py               # Functions for evaluating individual and ensemble models
│   └── metrics.py                # Metric calculation utilities
├── scripts/
│   ├── train_baseline.py         # Script to train the baseline model
│   ├── train_enhanced.py         # Script to train the enhanced model
│   └── evaluate_models.py        # Script to evaluate baseline, enhanced, and ensemble models
└── README.md                     # Project documentation (this file)
```

---

## Setup

### Prerequisites

- Python 3.7+
- Recommended: GPU with CUDA support for faster training

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/NLI_Project.git
   cd NLI_Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure that your `requirements.txt` includes necessary packages such
   as `torch`, `transformers`, `datasets`, `scikit-learn`, and `pandas`.

3. Download and prepare the NLI dataset:
   This project uses the FEVER dataset with semantic annotations. In `config/config.py`, specify the dataset source if
   needed:
   ```python
   CONFIG = {
       "dataset_name": "tommasobonomo/sem_augmented_fever_nli",
       # other configurations
   }
   ```

---

## Usage

### Training the Baseline Model

To train the baseline model, run:

```bash
python scripts/train_baseline.py
```

This script:

- Loads the training and validation datasets.
- Initializes and trains the baseline model.
- Saves the trained model as `baseline_model_checkpoint.pt` in the path specified by `CONFIG["save_path"]`.

### Training the Enhanced Model

To train the enhanced model, which uses WSD and SRL embeddings, run:

```bash
python scripts/train_enhanced.py
```

This script:

- Loads the training and validation datasets.
- Initializes and trains the enhanced model.
- Saves the trained model as `enhanced_model_checkpoint.pt`.

### Evaluating Models

To evaluate the baseline, enhanced, and ensemble models, run:

```bash
python scripts/evaluate_models.py
```

This script:

- Loads the test dataset and creates an adversarial version with data augmentation.
- Loads trained model checkpoints (or issues a warning if they’re missing).
- Evaluates the baseline, enhanced, and ensemble models on both the original and adversarial test sets.
- Saves the evaluation results as `evaluation_results.csv` in the save path.

---

## Configuration

All key configurations (e.g., model names, batch sizes, dataset paths) are set in `config/config.py`. Update this file
as needed for different models or dataset paths.

Example configuration:

```python
# config/config.py

CONFIG = {
    "model_name": "microsoft/deberta-base",
    "wsd_vocab_size": 10000,
    "srl_vocab_size": 50,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "max_length": 128,
    "save_path": "models/",
    "dataset_name": "tommasobonomo/sem_augmented_fever_nli"
}
```

---

## Detailed Module Explanations

### `data`

- **data_loader.py**: Functions for loading and splitting datasets, creating `DataLoader` objects.
- **augmentation.py**: Implements synonym and antonym substitution for data augmentation.
- **preprocess.py**: Prepares inputs for model training, including tokenization and WSD/SRL embedding processing.

### `models`

- **baseline_model.py**: Defines the baseline transformer model.
- **enhanced_model.py**: Defines the enhanced NLI model that integrates WSD and SRL embeddings.
- **utils.py**: Includes functions for saving/loading models and initializing baseline/enhanced models.

### `training`

- **train.py**: Contains `train_model`, which handles training for both the baseline and enhanced models, with options
  for saving checkpoints.

### `evaluation`

- **evaluate.py**: Functions for evaluating models, including ensemble evaluation by averaging logits from baseline and
  enhanced models.
- **metrics.py**: Implements `calculate_metrics`, which calculates accuracy, precision, recall, and F1-score for
  evaluation.

### `scripts`

- **train_baseline.py**: Trains the baseline model and saves it as a checkpoint.
- **train_enhanced.py**: Trains the enhanced model with WSD and SRL embeddings.
- **evaluate_models.py**: Evaluates the baseline, enhanced, and ensemble models on original and adversarial test sets,
  saving results to a CSV file.

---

## Example Workflow

1. **Train the Models**:
    - Run `train_baseline.py` and `train_enhanced.py` to train both models.

2. **Evaluate the Models**:
    - Run `evaluate_models.py` to evaluate the performance of each model on both original and adversarial datasets.
      Results will be displayed and saved in `evaluation_results.csv`.

3. **Analyze Results**:
    - Review the `evaluation_results.csv` for insights into each model’s performance. Compare metrics on the original
      and adversarial test sets to assess model robustness.

---

## Notes

- **Saving Checkpoints**: Models are saved after training. Be sure to check `CONFIG["save_path"]` to verify where models
  are saved.
- **Custom Configurations**: Update `config.py` to adjust hyperparameters, model names, and dataset paths for different
  experiments.

---
