"""Script for adversarial fine-tuning.
"""
import argparse

import torch
from torch import nn
from tqdm.auto import tqdm

from utils import *
from data import *
from models import *


def train_threat_model(train_loader, val_loader, model, optimizer, device, args):
    model.to(device)
    train_progress_bar = tqdm(range(len(train_loader)))
    val_progress_bar = tqdm(range(len(val_loader)))

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        
        step = 0
        training_loss = 0
        model.train()
        
        train_progress_bar.refresh()
        train_progress_bar.reset()

        for batch in train_loader:
            step += 1
        
            sentence_embedding = batch['sentence_embedding'].to(device)
            aux_label = batch['aux_label'].to(device)

            output, loss = model(sentence_embedding, aux_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss

            train_progress_bar.update(1)

        training_loss /= step

        print(f"epoch {epoch + 1} train loss: {training_loss:.4f}")


        if (epoch + 1) % args.val_interval == 0:
            step = 0
            val_loss = 0
            
            model.eval()

            val_progress_bar.refresh()
            val_progress_bar.reset()

            with torch.no_grad():
                for batch in val_loader:
                    step += 1
                    sentence_embedding = batch['sentence_embedding'].to(device)
                    aux_label = batch['aux_label'].to(device)
                    output, loss = model(sentence_embedding, aux_label)
                    val_loss += loss
                    val_progress_bar.update(1)
            val_loss /= step
            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Visualize last batch
            one_hot_labels = aux_label.cpu()[:10]
            one_hot_predictions = output.cpu()[:10]
            for one_hot_label, one_hot_prediction in zip(one_hot_labels, one_hot_predictions):
                id_label = torch.where(one_hot_label == 1)[0]
                id_pred = torch.where(one_hot_prediction == 1)[0]
                decoded_label = aux_tokenizer.decode(id_label)
                decoded_pred = aux_tokenizer.decode(id_pred)
                print(f"Ground Truth Set: {decoded_label}, Pred Set: {decoded_pred}")


    train_progress_bar.close()
    val_progress_bar.close()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_interval", type=int, default=1)

    args = parser.parse_args()

    train_dataset, train_loader, val_dataset, val_loader = load_aux_data("data/quora_duplicate_questions.tsv", args.sample_size, args.batch_size, args.batch_size)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = MultiSetInversionModel(emb_dim=bert_aux_config.hidden_size, output_size=bert_aux_config.vocab_size, 
                                   steps=32, device=device)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train threat
    train_threat_model(train_loader, val_loader, model, optimizer, device, args)


