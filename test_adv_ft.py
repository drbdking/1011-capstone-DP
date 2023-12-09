"""Script for test of adversarial fine-tuning.
"""
import argparse

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import BertModel, BertConfig
import torch
from torch import nn
from tqdm.auto import tqdm


from utils import *
from data import *
from models import *


def test(test_loader, bert_model, cls_model):
    step = 0

    bert_model.eval()
    cls_model.eval()
    test_progress_bar = tqdm(range(len(test_loader)))

    conf_mat = ConfusionMatrix(n_classes=2, device=device)

    with torch.no_grad():
        for batch in test_loader:
            step += 1
            # Inference, everything can be reused

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            hidden_states = bert_model(input_ids, attention_mask)['last_hidden_state']  # remove token_type_ids

            sentence_embedding = torch.mean(hidden_states, dim=1)
            cls_output = cls_model(sentence_embedding)
            label = batch['label'].to(device)
            conf_mat += torch.argmax(cls_output, dim=1), label

            test_progress_bar.update(1)
            
    tn, fn, fp, tp = conf_mat.value
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    print(f"acc: {acc}, f1 score: {f1_score}, precision: {precision}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    test_dataset, test_loader = load_test_data(tsv_path="data/qqp_embedding_test.tsv")

    # Bert Model
    bert_config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = BertModel.from_pretrained("bert-base-uncased", config=bert_config)
    cls_model = BinaryClassificationHead(input_size=bert_config.hidden_size)
    bert_model.to(device)
    cls_model.to(device)

    # Load
    model_state_dict = torch.load(args.model_path, map_location='cpu')
    bert_model.load_state_dict(model_state_dict['base_state_dict'])
    cls_model.load_state_dict(model_state_dict['cls_state_dict'])
    
    # Test
    test(test_loader, bert_model, cls_model)