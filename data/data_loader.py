# data/data_loader.py
from datasets import load_dataset
from torch.utils.data import DataLoader
from config.config import CONFIG


def load_dataset_splits():
    dataset = load_dataset(CONFIG["dataset_name"])
    return dataset['train'], dataset['validation'], dataset['test']


def create_data_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
