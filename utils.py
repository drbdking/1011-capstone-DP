"""Split dataset, load and process data
"""
import string
import json
from collections import defaultdict


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

# stop_words = stopwords.words('english')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bert_aux_config = BertConfig.from_pretrained("bert-base-uncased")
# bert_aux_model = BertModel.from_pretrained("bert-base-uncased")
# aux_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_aux_model.to(device)

def tokenize_func(input):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(input['question1'], input['question2'], padding="max_length", truncation=True)  # max_len

# def remove_stopwords_punc(sentence):
#     t = filter(lambda x: x.lower() not in stop_words, sentence.split(' '))
#     return ' '.join(map(lambda x: x.strip(string.punctuation), t))

# def preprocess_func_aux(input):
#     # Batch processing is not implemented here due to potential variable length problem
#     # Note: for convenience it is uncased

#     input['question1'] = remove_stopwords_punc(input['question1'])
#     input['question2'] = remove_stopwords_punc(input['question2'])

#     # Get sentence embedding
#     encoding = aux_tokenizer(input['question1'], input['question2'], truncation=True, add_special_tokens=False)  # Remove cls and sep
#     input_ids = torch.Tensor(encoding['input_ids']).to(torch.int64).reshape(1, -1).to(device)
#     token_type_ids = torch.Tensor(encoding['token_type_ids']).to(torch.int64).reshape(1, -1).to(device)
#     attention_mask = torch.Tensor(encoding['attention_mask']).to(torch.int64).reshape(1, -1).to(device)
#     bert_output = bert_aux_model(input_ids, attention_mask, token_type_ids)['last_hidden_state']
#     sentence_embedding = torch.mean(bert_output, dim=1)  # Careful with dim, the first dim is batch, second is seq_len
#     input['sentence_embedding'] = sentence_embedding.reshape(-1).cpu()

#     # Get label for multi set
#     input_ids_one_hot = F.one_hot(torch.Tensor(encoding['input_ids']).to(torch.int64), num_classes=bert_aux_config.vocab_size)  # size = seq_len * vocab_size
#     aux_label = torch.sum(input_ids_one_hot, dim=0).int()  # Careful with dim, first is seq_len
#     input['aux_label'] = aux_label.to(torch.bool)
#     return input

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
    
class ResultRecorder(object):
    def __init__(self, train_mode, params) -> None:
        self.record = defaultdict(list)
        self.train_mode = train_mode
        self.params = [str(x) for x in params]

    def __setitem__(self, key, value):
        self.record[key].append(value)

    def save(self, output_dir):
        with open(output_dir + self.train_mode + "_" + "_".join(self.params) + ".json", "w") as fp:
            json.dump(self.record, fp)


    

