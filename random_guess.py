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
    parser.add_argument("--times", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--test_sample_size", type=int, default=10000)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str, default='BERT')

    args = parser.parse_args()

    print("----------Start Loading Model----------")

    load_model(args.model_path, args.model_type)

    print("----------Start Loading Data----------")

    test_dataset, test_loader = load_aux_test_data("data/qqp_threat_test.tsv", args.test_sample_size, args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("----------Start Random Guessing----------")

    # Random Guess
    loss, precision, recall, f1 = [], [], [], []

    for i in tqdm(range(args.times)):
        model = MultiSetInversionModel(emb_dim=768, output_size=30522, steps=32, device=device)
        model.to(device)
        test_loss, test_precision, test_recall, test_f1 = test_threat_model(test_loader, model, device)
        loss.append(test_loss)
        precision.append(test_precision)
        recall.append(test_recall)
        f1.append(test_f1)

    loss_tensor = torch.tensor(loss)
    precision_tensor = torch.tensor(precision)
    recall_tensor = torch.tensor(recall)
    f1_tensor = torch.tensor(f1)

    loss_mean, loss_std = loss_tensor.mean(), loss_tensor.std()
    precision_mean, precision_std = precision_tensor.mean(), precision_tensor.std()
    recall_mean, recall_std = recall_tensor.mean(), recall_tensor.std()
    f1_mean, f1_std = f1_tensor.mean(), f1_tensor.std()

    print(f'test loss, mean: {loss_mean:4f}, std: {loss_std:4f}')
    print(f'test precision, mean: {precision_mean:4f}, std: {precision_std:4f}')
    print(f'test recall, mean: {recall_mean:4f}, std: {recall_std:4f}')
    print(f'test f1, mean: {f1_mean:4f}, std: {f1_std:4f}')





