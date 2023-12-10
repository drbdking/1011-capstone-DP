"""Script for adversarial fine-tuning.
"""
import argparse

import torch
from torch import nn
from tqdm.auto import tqdm

from utils import *
from data import *
from models import *


def get_tp_fp_fn_metrics(pred, labels):
    tp = torch.sum((pred == 1) & (labels == 1))
    fp = torch.sum((pred == 1) & (labels == 0))
    fn = torch.sum((pred == 0) & (labels == 1))
    return tp.item(), fp.item(), fn.item()


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
            tp_total, fp_total, fn_total, val_loss = 0, 0, 0, 0
            
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

            # Calculate evaluation metrics on the last batch
            one_hot_labels = aux_label.cpu()
            one_hot_predictions = output.cpu()
            for one_hot_label, one_hot_prediction in zip(one_hot_labels, one_hot_predictions):
                tp, fp, fn = get_tp_fp_fn_metrics(one_hot_prediction, one_hot_label)
                tp_total += tp
                fp_total += fp
                fn_total += fn
            precision = tp_total / (tp_total + fp_total)
            recall = tp_total / (tp_total + fn_total)
            f1 = 2 * precision * recall / (precision + recall)

            print(f"epoch {epoch + 1}, val loss: {val_loss:.4f}, val precision: {precision:.4f}, val recall: {recall:.4f}, val f1: {f1:.4f}")

            # Visualize last batch using 10 examples
            one_hot_labels = aux_label.cpu()[:10]
            one_hot_predictions = output.cpu()[:10]
            for one_hot_label, one_hot_prediction in zip(one_hot_labels, one_hot_predictions):
                id_label = torch.where(one_hot_label == 1)[0]
                id_pred = torch.where(one_hot_prediction == 1)[0]
                decoded_label = aux_tokenizer.decode(id_label)
                decoded_pred = aux_tokenizer.decode(id_pred)
                print(f"Ground Truth Set: {decoded_label}")
                print(f"Prediction Set: {decoded_pred}")
                print("----------")


    train_progress_bar.close()
    val_progress_bar.close()
    return 


def test_threat_model(test_loader, model, device):

    model.eval()
    test_progress_bar = tqdm(range(len(test_loader)))
    tp_total, fp_total, fn_total, test_loss, step = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch in test_loader:
            step += 1
            sentence_embedding = batch['sentence_embedding'].to(device)
            aux_label = batch['aux_label'].to(device)
            output, loss = model(sentence_embedding, aux_label)
            test_loss += loss
            test_progress_bar.update(1)
    test_loss /= step

    one_hot_labels = aux_label.cpu()
    one_hot_predictions = output.cpu()
    for one_hot_label, one_hot_prediction in zip(one_hot_labels, one_hot_predictions):
        tp, fp, fn = get_tp_fp_fn_metrics(one_hot_prediction, one_hot_label)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"test loss: {test_loss:.4f}, test precision: {precision:.4f}, test recall: {recall:.4f}, test f1: {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--mask_magnitude", type=float, default=0)
    parser.add_argument("--test_sample_size", type=int, default=2000)

    args = parser.parse_args()

    train_dataset, train_loader, val_dataset, val_loader = load_aux_data("data/qqp_threat_train.tsv", args.sample_size, args.batch_size, args.batch_size)
    test_dataset, test_loader = load_aux_test_data("data/qqp_threat_test.tsv", args.test_sample_size, args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = MultiSetInversionModel(emb_dim=bert_aux_config.hidden_size, output_size=bert_aux_config.vocab_size, 
                                   steps=32, device=device, mask_magnitude=args.mask_magnitude)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("----------Start Training----------")

    # Train threat model
    train_threat_model(train_loader, val_loader, model, optimizer, device, args)

    print("----------Start Testing----------")

    # Test threat model
    test_threat_model(test_loader, model, device)



