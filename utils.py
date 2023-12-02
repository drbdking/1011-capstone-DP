"""Split dataset, load and process data
"""
import torch
import pandas as pd
from transformers import AutoTokenizer


def tokenize_func(input):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(input['question1'], input['question2'], padding="max_length", truncation=True)



class ConfusionMatrix:
    def __init__(self, n_classes, device):
        self._matrix = torch.zeros(n_classes * n_classes).to(device)
        self._n = n_classes

    def __add__(self, other):
        if isinstance(other, ConfusionMatrix):
            self._matrix.add_(other._matrix)
        elif isinstance(other, tuple):
            self.update(*other)
        else:
            raise NotImplemented
        return self

    def update(self, prediction: torch.tensor, label: torch.tensor):
        conf_data = (prediction * self._n + label).int()
        conf = conf_data.bincount(minlength=self._n * self._n)
        self._matrix.add_(conf)

    @property
    def value(self):
        return self._matrix.cpu().tolist()
