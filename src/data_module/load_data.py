"""
- classes: Dataset, dataloader...
- functions: load_data_to_pd, collate_fn, setup_dataloader
"""

import json
from torch.utils.data import DataLoader
import pandas as pd

import datasets
from datasets import load_dataset
from src.data_module.DataCollator import DataCollator

def do_train_dev_test_split(raw_datasets, train_dev_test_split_ratio, seed):
    [train_frac, dev_frac, test_frac] = train_dev_test_split_ratio
    train_size = int(train_frac * len(raw_datasets))
    dev_size = int(dev_frac * len(raw_datasets))
    test_size = len(raw_datasets) - train_size - dev_size
    train_dataset, dev_dataset, test_dataset = datasets.Dataset.train_test_split(
        raw_datasets,
        test_size=dev_size + test_size,
        seed=seed
    ).values()
    dev_dataset, test_dataset = datasets.Dataset.train_test_split(
        dev_dataset,
        test_size=test_size,
        seed=seed
    ).values()
    raw_datasets = datasets.DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset,
        "predict": test_dataset
    })
    return raw_datasets


def load_data_from_hf(file_or_dataset_name, train_dev_test_split_ratio, seed, cache_dir):
    """processes data into huggingface dataset"""
    if not file_or_dataset_name.endswith(".csv") and not (file_or_dataset_name.endswith(".json") or file_or_dataset_name.endswith(".jsonl")):
        raw_datasets = load_dataset(file_or_dataset_name, cache_dir=cache_dir)
    elif file_or_dataset_name.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=file_or_dataset_name,
            cache_dir=cache_dir,
        )
    elif file_or_dataset_name.endswith(".json") or file_or_dataset_name.endswith(".jsonl"):
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=file_or_dataset_name,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"unknown dataset format {file_or_dataset_name}")
    
    if len(raw_datasets) == 1:
        # if there is only one split, do train/dev/test split
        raw_datasets = do_train_dev_test_split(raw_datasets[list(raw_datasets.keys())[0]], train_dev_test_split_ratio, seed)
    return raw_datasets



def load_data_to_pd(file_or_dataset_name, return_df_only=False):
    if file_or_dataset_name.endswith("jsonl"):
        with open(file_or_dataset_name, "r") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
            if return_df_only:
                return pd.DataFrame(data)
            return datasets.Dataset.from_pandas(pd.DataFrame(data))
    pass


def setup_dataloader(input_datasets, batch_size, tokenizer):
    """

    :param input_datasets: dictionary of datasets (train, eval, predict)
    :param batch_size: encoded test dataset
    :return:
    """
    dataloaders = {}
    for split in ["train", "eval", "predict"]:
        if split not in input_datasets:
            dataloaders[split] = None
        else:
            dataloaders[split] = DataLoader(input_datasets[split], shuffle=split == "train", batch_size=batch_size, collate_fn=DataCollator(tokenizer))
    return dataloaders


def load_gsm8k_compressed(data_path, train_dev_test_split_ratio, seed):
    """Load GSM8K compressed dataset from JSON file.

    The dataset contains reasoning traces with <COMP> tokens marking
    compression points. Format:
    {
        "question": "...",
        "reasoning_with_compression": "... <COMP> ... <COMP> ...",
        "answer": "42",
        "original_reasoning": "..."
    }

    Args:
        data_path: Path to the JSON file (e.g., data/gsm8k_compressed_train.json)
        train_dev_test_split_ratio: List of [train, val, test] fractions
        seed: Random seed for shuffling

    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    import random

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Shuffle data
    random.seed(seed)
    random.shuffle(data)

    # Calculate split sizes
    n = len(data)
    train_frac, val_frac, test_frac = train_dev_test_split_ratio
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Create splits
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Convert to HuggingFace Dataset
    return datasets.DatasetDict({
        'train': datasets.Dataset.from_list(train_data),
        'validation': datasets.Dataset.from_list(val_data),
        'test': datasets.Dataset.from_list(test_data)
    })