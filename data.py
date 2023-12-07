import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import pandas as pd


from utils import *


def split_data(df, ratios, shuffle=True, random_state=42):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        ratios (list): _description_
        shuffle (bool, optional): _description_. Defaults to True.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        list: list of DataFrames split according to given ratios
    """
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return np.split(df, np.cumsum((np.array(ratios[:-1]) * len(df)).astype(int)))
    
    
def load_data(tsv_path, train_batch_size, val_batch_size):
    df = pd.read_csv(tsv_path, sep = '\t')
    data = {'question1': df['question1'].astype(str).tolist(), 
            'question2': df['question2'].astype(str).tolist(), 
            'label': df['is_duplicate'].tolist()}
    qqp_dataset = Dataset.from_dict(data)
    qqp_dataset = qqp_dataset.shuffle()
    qqp_dataset = qqp_dataset.select(range(10000))
    qqp_dataset = qqp_dataset.map(tokenize_func, batched=True, load_from_cache_file=False)
    qqp_dataset = qqp_dataset.remove_columns(['question1', 'question2'])
    train_dataset, val_dataset = qqp_dataset.train_test_split(test_size=0.2).values()
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    return train_dataset, train_loader, val_dataset, val_loader

def load_aux_data(tsv_path, sample_size, train_batch_size, val_batch_size):
    df = pd.read_csv(tsv_path, sep = '\t')
    data = {'question1': df['question1'].astype(str).tolist(), 
            'question2': df['question2'].astype(str).tolist(), 
            }
    qqp_dataset = Dataset.from_dict(data)
    qqp_dataset = qqp_dataset.shuffle()
    qqp_dataset = qqp_dataset.select(range(sample_size))
    qqp_dataset = qqp_dataset.map(preprocess_func_aux, load_from_cache_file=False)
    qqp_dataset = qqp_dataset.remove_columns(['question1', 'question2'])

    train_dataset, val_dataset = qqp_dataset.train_test_split(test_size=0.2).values()
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    return train_dataset, train_loader, val_dataset, val_loader