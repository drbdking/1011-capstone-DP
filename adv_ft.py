"""Script for adversarial fine-tuning.
"""
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import BertModel, BertConfig
import torch


from utils import *

def train_adv(num_of_epochs, embedding_model_lr, cls_lr, adv_lr):
    # A few thoughts about tunable hyperparameters:
    # separate learning rate and num of training epochs for each model (classification, embedding, adversary)
    # The input (tgt) of adversary is the embedding / hidden_states[0] from Bert, the label is a tensor of input token ids 
    return 