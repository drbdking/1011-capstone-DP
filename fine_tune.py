"""Script for adversarial fine-tuning.
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


def train_embedding(train_loader, val_loader, embedding_dict, device, args):
    # A few thoughts about tunable hyperparameters:
    # separate learning rate and num of training epochs for each model (classification, embedding, adversary)
    # The input (tgt) of adversary is the embedding / hidden_states[0] from Bert, the label is a tensor of input token ids
    # epoch iteration
    embedding_dict['base_model'].to(device)
    embedding_dict['classifier'].to(device)
    train_progress_bar = tqdm(range(len(train_loader)))
    val_progress_bar = tqdm(range(len(val_loader)))

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        
        step = 0
        embedding_train_cls_loss = 0
        embedding_dict['base_model'].train()
        embedding_dict['classifier'].train()
        
        train_progress_bar.refresh()
        train_progress_bar.reset()

        for batch in train_loader:
            step += 1
        
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Train embedding
            # Mean pool the hidden states from Bert model and feed into classifier
            hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state']
            sentence_embedding = torch.mean(hidden_states, dim=1)
            # sentence_embedding = hidden_states
            cls_output = embedding_dict['classifier'](sentence_embedding)
            label = batch['label'].to(device)
            cls_loss = embedding_dict['loss_function'](cls_output, label)
            cls_loss.backward()
            embedding_dict['optimizer'].step()
            embedding_dict['optimizer'].zero_grad()
            embedding_train_cls_loss += cls_loss.item()

            train_progress_bar.update(1)

        embedding_train_cls_loss /= step

        print(f"epoch {epoch + 1} average embedding train cls loss: {embedding_train_cls_loss:.4f}")


        if (epoch + 1) % args.val_interval == 0:
            step = 0
            embedding_val_cls_loss = 0
            conf_mat = ConfusionMatrix(n_classes=2, device=device)
            
            embedding_dict['base_model'].eval()
            embedding_dict['classifier'].eval()

            val_progress_bar.refresh()
            val_progress_bar.reset()

            with torch.no_grad():
                for batch in val_loader:
                    step += 1
                    # Inference, everything can be reused
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state']
                    sentence_embedding = torch.mean(hidden_states, dim=1)
                    # sentence_embedding = hidden_states
                    cls_output = embedding_dict['classifier'](sentence_embedding)
                    label = batch['label'].to(device)
                    conf_mat += torch.argmax(cls_output, dim=1), label
                    cls_loss = embedding_dict['loss_function'](cls_output, label)
                    embedding_val_cls_loss += cls_loss.item()
                    val_progress_bar.update(1)
                    
            embedding_val_cls_loss /= step
            tn, fn, fp, tp = conf_mat.value
            print(f"epoch {epoch + 1} average embedding val cls loss: {embedding_val_cls_loss:.4f}, acc: {(tp+tn) / (tp + tn + fp + fn)}, f1 score: {2 * tp / (2 * tp + fp + fn)}, precision: {tp / (tp + fp)}")

    train_progress_bar.close()
    val_progress_bar.close()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--downsample", type=float, default=0.25)
    parser.add_argument("--val_interval", type=int, default=1)

    args = parser.parse_args()

    train_dataset, train_loader, val_dataset, val_loader = load_data("data/qqp_embedding_train.tsv", args.downsample, args.batch_size, args.batch_size)
    
    # Device
    device = torch.device("cuda:0")

    # Bert Model
    bert_config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
    bert_model = BertModel.from_pretrained("bert-base-cased", config=bert_config)
    cls_model = BinaryClassificationHead(input_size=bert_config.hidden_size)
    embedding_optimizer = torch.optim.Adam([
        {"params": bert_model.parameters(), "lr": 1e-5},
        {"params": cls_model.parameters(), "lr": 1e-5},
        ])

    embedding_dict = {
        'base_model': bert_model,
        'classifier': cls_model,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': embedding_optimizer
    }

    train_embedding(train_loader, val_loader, embedding_dict, device, args)


