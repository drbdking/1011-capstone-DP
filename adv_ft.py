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


def train_adv(train_loader, val_loader, adv_dict, embedding_dict, device, args):
    # A few thoughts about tunable hyperparameters:
    # separate learning rate and num of training epochs for each model (classification, embedding, adversary)
    # The input (tgt) of adversary is the embedding / hidden_states[0] from Bert, the label is a tensor of input token ids
    # epoch iteration
    adv_dict['model'].to(device)
    embedding_dict['base_model'].to(device)

    for epoch in range(args.num_epochs):
        # train adv, get embedding (no grad), hidden state and label (input token ids)
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        
        step = 0
        train_loss = 0
        adv_dict['model'].train()
        train_progress_bar = tqdm(range(len(train_loader)))
        for batch in train_loader:
            step += 1
            # Input of adv model, no grad
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = embedding_dict['base_model'].embeddings(input_ids, token_type_ids).detach()
            hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state'].detach()
            output = adv_dict['model'](embeddings, hidden_states)
            # CrossEntropy loss target format (batch size, num_classes, d1, d2 ...)
            output = torch.transpose(output, 1, 2)
            loss = adv_dict['loss_function'](output, input_ids)
            loss.backward()
            adv_dict['optimizer'].step()
            adv_dict['optimizer'].zero_grad()
            train_loss += loss.item()
            train_progress_bar.update(1)
        train_loss /= step
        print(f"epoch {epoch + 1} average train loss: {train_loss:.4f}")
        train_progress_bar.close()

        if (epoch + 1) % args.val_interval == 0:
            val_loss = 0
            step = 0
            adv_dict['model'].eval()
            val_progress_bar = tqdm(range(len(val_loader)))
            with torch.no_grad():
                for batch in val_loader:
                    step += 1
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    embeddings = embedding_dict['base_model'].embeddings(input_ids, token_type_ids).detach()
                    hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state'].detach()
                    output = adv_dict['model'](embeddings, hidden_states)
                    output = torch.transpose(output, 1, 2)
                    loss = adv_dict['loss_function'](output, input_ids)
                    val_loss += loss.item()
                    val_progress_bar.update(1)
            val_loss /= step
            print(f"epoch {epoch + 1} average val loss: {val_loss:.4f}")
            val_progress_bar.close()
    # eval adv
    # train bert model, get input ids, attention mask, token type ids
    # eval bert
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_interval", type=int, default=1)

    args = parser.parse_args()

    train_dataset, train_loader, val_dataset, val_loader = load_data("data/quora_duplicate_questions.tsv", args.batch_size, args.batch_size)
    
    # Device
    device = torch.device("cuda:0")

    # Bert Model
    bert_config = BertConfig.from_pretrained("bert-base-cased", output_hidden_states=True)
    bert_model = BertModel.from_pretrained("bert-base-cased", config=bert_config)

    embedding_dict = {
        'base_model': bert_model
    }

    # Adv model
    adv_model = AdversarialDecoder(tgt_vocab_size=bert_config.vocab_size, device=device)
    optimizer = torch.optim.Adam(adv_model.parameters(), 1e-4)
    adv_dict = {
        'model': adv_model,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': optimizer
    }

    train_adv(train_loader, val_loader, adv_dict, embedding_dict, device, args)


