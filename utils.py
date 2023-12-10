"""Split dataset, load and process data
"""
import string

import torch
from torch.nn import functional as F
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import BertModel, BertConfig
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_aux_config = BertConfig.from_pretrained("bert-base-uncased")
bert_aux_model = BertModel.from_pretrained("bert-base-uncased")
aux_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_aux_model.to(device)

def tokenize_func(input):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(input['question1'], input['question2'], padding="max_length", truncation=True)

def remove_stopwords_punc(sentence):
    t = filter(lambda x: x.lower() not in stop_words, sentence.split(' '))
    return ' '.join(map(lambda x: x.strip(string.punctuation), t))

def preprocess_func_aux(input):
    # Batch processing is not implemented here due to potential variable length problem
    # Note: for convenience it's uncased

    # Get sentence embedding
    encoding = aux_tokenizer(input['question1'], input['question2'], padding="max_length", truncation=True)  # Add padding
    input_ids = torch.Tensor(encoding['input_ids']).to(torch.int64).reshape(1, -1).to(device)
    attention_mask = torch.Tensor(encoding['attention_mask']).to(torch.int64).reshape(1, -1).to(device)
    bert_output = bert_aux_model(input_ids, attention_mask)['last_hidden_state']
    sentence_embedding = torch.mean(bert_output, dim=1)  # Careful with dim, the first dim is batch, second is seq_len
    input['sentence_embedding'] = sentence_embedding.reshape(-1).cpu()

    # Get label for multi set
    input['question1'] = remove_stopwords_punc(input['question1'])
    input['question2'] = remove_stopwords_punc(input['question2'])
    label_encoding = aux_tokenizer(input['question1'], input['question2'], padding="max_length", truncation=True, add_special_tokens=False)  # Remove cls and sep; Add padding
    input_ids_one_hot = F.one_hot(torch.Tensor(label_encoding['input_ids']).to(torch.int64), num_classes=bert_aux_config.vocab_size)  # size = seq_len * vocab_size
    aux_label = torch.sum(input_ids_one_hot, dim=0).int()  # Careful with dim, first is seq_len
    input['aux_label'] = aux_label.to(torch.bool)
    return input


