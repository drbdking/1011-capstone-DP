"""Split dataset, load and process data
"""
import pandas as pd
from transformers import AutoTokenizer


def tokenize_func(input):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(input['question1'], input['question2'], padding='max_length', truncation=True)

