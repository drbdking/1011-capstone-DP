"""Script for adversarial fine-tuning.
"""
import argparse
import os
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

    training_losses = []
    validation_losses = []

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")

        # Training Loss
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
        training_losses.append(training_loss)

        # Validation Loss
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
        validation_losses.append(val_loss)

        print(f"epoch {epoch + 1}, training loss: {training_loss:.4f}, val loss: {val_loss:.4f}")


        # Evaluation Metrics
        if (epoch + 1) % args.val_interval == 0:
            tp_total, fp_total, fn_total = 0, 0, 0

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
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

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

    return model, training_losses, validation_losses


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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return test_loss, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--mask_magnitude", type=float, default=0)
    parser.add_argument("--test_sample_size", type=int, default=10000)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str, default='ADV')

    args = parser.parse_args()

    print("----------Start Loading Model----------")

    load_model(args.model_path, args.model_type)

    print("----------Start Loading Data----------")

    train_dataset, train_loader, val_dataset, val_loader = load_aux_data("data/qqp_threat_train.tsv", args.sample_size, args.batch_size, args.batch_size)
    test_dataset, test_loader = load_aux_test_data("data/qqp_threat_test.tsv", args.test_sample_size, args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = MultiSetInversionModel(emb_dim=768, output_size=30522,
                                   steps=32, device=device, mask_magnitude=args.mask_magnitude)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("----------Start Training----------")

    # Train threat model
    trained_model, training_losses, validation_losses = train_threat_model(train_loader, val_loader, model, optimizer, device, args)

    training_losses = [loss.item() for loss in training_losses]
    validation_losses = [loss.item() for loss in validation_losses]

    print("----------Start Testing----------")

    # Test threat model
    test_loss, test_precision, test_recall, test_f1 = test_threat_model(test_loader, trained_model, device)

    print("----------Save Results----------")

    model_path = args.model_path
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]
    results_file_path = os.path.join("results", model_name + ".txt")

    if not os.path.exists('results'):
        os.makedirs('results')

    with open(results_file_path, 'w') as log_file:

        log_file.write("Training Loss:\n")
        training_loss_str = ",".join([f"{loss:.4f}" for loss in training_losses])
        log_file.write(training_loss_str + "\n")

        log_file.write("\nValidation Loss:\n")
        validation_loss_str = ",".join([f"{loss:.4f}" for loss in validation_losses])
        log_file.write(validation_loss_str + "\n")

        log_file.write("\nTest Metrics:\n")
        log_file.write(f"Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}\n")

    print("----------Print Results----------")

    print('Training Loss:')
    print(training_losses)
    print(' ')
    print('Validation Loss:')
    print(validation_losses)
    print(' ')
    print('Test Metrics:')
    print(f"Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}")






