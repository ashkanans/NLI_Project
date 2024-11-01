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
