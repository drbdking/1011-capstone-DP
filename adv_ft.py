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
    embedding_dict['classifier'].to(device)
    train_progress_bar = tqdm(range(len(train_loader)))
    val_progress_bar = tqdm(range(len(val_loader)))

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        step = 0
        adv_train_loss = 0
        embedding_train_adv_loss = 0
        embedding_train_cls_loss = 0
        adv_dict['model'].train()
        embedding_dict['base_model'].train()
        embedding_dict['classifier'].train()
        
        train_progress_bar.refresh()
        train_progress_bar.reset()

        for batch in train_loader:
            step += 1
            # Train adv, get embedding (no grad), hidden state and label (input token ids)
            # Zero grad to eliminate embedding model training gradient 
            adv_dict['optimizer'].zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Input of adv model, no grad by using detach()
            # It is also possible to keep the grad and do zero grad when training embedding model
            embeddings = embedding_dict['base_model'].embeddings(input_ids, token_type_ids)
            embeddings_no_grad = embeddings.detach()
            hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state']
            hidden_states_no_grad = hidden_states.detach()
            adv_output = adv_dict['model'](embeddings_no_grad, hidden_states_no_grad)

            # CrossEntropy loss target format (batch size, num_classes, d1, d2 ...)
            adv_output = torch.transpose(adv_output, 1, 2)
            adv_loss = adv_dict['loss_function'](adv_output, input_ids)
            adv_loss.backward()
            adv_dict['optimizer'].step()
            adv_dict['optimizer'].zero_grad()
            adv_train_loss += adv_loss.item()

            train_progress_bar.update(1)

        step = 0
        train_progress_bar.refresh()
        train_progress_bar.reset()

        for batch in train_loader:
            # Train embedding
            step += 1
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            embeddings = embedding_dict['base_model'].embeddings(input_ids, token_type_ids)
            hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state']
            
            # Mean pool the hidden states from Bert model and feed into classifier
            sentence_embedding = torch.mean(hidden_states, dim=1)
            cls_output = embedding_dict['classifier'](sentence_embedding)
            label = batch['label'].to(device)
            # Embedding loss = cls loss - alpha * adv loss
            # Need to regenerate adv output and adv loss since we updated adv parameters
            # Embeddings and hidden states can be reused since we didn't update embedding model when training adversary 
            adv_output = adv_dict['model'](embeddings, hidden_states)
            adv_output = torch.transpose(adv_output, 1, 2)
            adv_loss = adv_dict['loss_function'](adv_output, input_ids)
            cls_loss = embedding_dict['loss_function'](cls_output, label)
            # emebdding_loss = cls_loss - 1.5 * adv_loss
            embedding_loss = -adv_loss
            embedding_loss.backward()
            embedding_dict['optimizer'].step()
            embedding_dict['optimizer'].zero_grad()
            embedding_train_adv_loss += adv_loss.item()
            embedding_train_cls_loss += cls_loss.item()

            train_progress_bar.update(1)

        adv_train_loss /= step
        embedding_train_adv_loss /= step
        embedding_train_cls_loss /= step
        print(f"epoch {epoch + 1} average adv train loss: {adv_train_loss:.4f}")
        print(f"epoch {epoch + 1} average embedding train adv loss: {embedding_train_adv_loss:.4f}, average embedding train cls loss: {embedding_train_cls_loss:.4f}")


        if (epoch + 1) % args.val_interval == 0:
            step = 0
            adv_val_loss = 0
            embedding_val_cls_loss = 0
            
            adv_dict['model'].eval()
            embedding_dict['base_model'].eval()
            embedding_dict['classifier'].eval()

            val_progress_bar.refresh()
            val_progress_bar.reset()

            with torch.no_grad():
                for batch in val_loader:
                    step += 1
                    # Inference, everything can be reused
                    # Validate adv
                    input_ids = batch['input_ids'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    embeddings = embedding_dict['base_model'].embeddings(input_ids, token_type_ids)
                    hidden_states = embedding_dict['base_model'](input_ids, attention_mask, token_type_ids)['last_hidden_state']
                    adv_output = adv_dict['model'](embeddings, hidden_states)
                    adv_output = torch.transpose(adv_output, 1, 2)
                    adv_loss = adv_dict['loss_function'](adv_output, input_ids)
                    adv_val_loss += adv_loss.item()

                    # Validate embedding
                    sentence_embedding = torch.mean(hidden_states, dim=1)
                    cls_output = embedding_dict['classifier'](sentence_embedding)
                    label = batch['label'].to(device)
                    cls_loss = embedding_dict['loss_function'](cls_output, label)
                    embedding_val_cls_loss += cls_loss.item()

                    val_progress_bar.update(1)
                    
                # Decode the last batch
                # output_ids = torch.argmax(adv_output, dim=1).detach().cpu()
                # original_seq = tokenizer.batch_decode(input_ids.cpu())
                # pred_seq = tokenizer.batch_decode(output_ids)
                # for ref, pred in zip(original_seq, pred_seq):
                #     print(f"Original input: {ref} \nAdv output: {pred}")

            adv_val_loss /= step
            embedding_val_cls_loss /= step
            print(f"epoch {epoch + 1} average val loss: {adv_val_loss:.4f}")
            print(f"epoch {epoch + 1} average embedding val cls loss: {embedding_val_cls_loss:.4f}")

    train_progress_bar.close()
    val_progress_bar.close()
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

    # Adv model
    adv_model = AdversarialDecoder(tgt_vocab_size=bert_config.vocab_size, device=device)
    adv_optimizer = torch.optim.Adam(adv_model.parameters(), 5e-5)
    adv_dict = {
        'model': adv_model,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': adv_optimizer
    }

    train_adv(train_loader, val_loader, adv_dict, embedding_dict, device, args)


