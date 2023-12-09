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


def train_adv(train_loader, val_loader, adv_dict, embedding_dict, recorder, device, args):
    # A few thoughts about tunable hyperparameters:
    # separate learning rate and num of training epochs for each model (classification, embedding, adversary)
    # The input (tgt) of adversary is the embedding / hidden_states[0] from Bert, the label is a tensor of input token ids
    # epoch iteration
    param_lst = [args.downsample, args.adv_mode, args.num_epochs, args.warmup_epochs, args.learning_rate, args.adv_learning_rate, args.alpha, args.adv_interval]    
    param_lst = [str(x) for x in param_lst]

    adv_dict['model'].to(device)
    embedding_dict['base_model'].to(device)
    embedding_dict['classifier'].to(device)
    total_train_step = len(train_loader)
    total_val_step = len(val_loader)
    # train_progress_bar = tqdm(range(len(train_loader)))
    # val_progress_bar = tqdm(range(len(val_loader)))

    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        
        step = 0
        adv_train_loss = 0
        embedding_train_adv_loss = 0
        embedding_train_cls_loss = 0
        adv_dict['model'].train()
        embedding_dict['base_model'].train()
        embedding_dict['classifier'].train()
        
        # train_progress_bar.refresh()
        # train_progress_bar.reset()

        if args.adv_mode == 0:
            # This is the common practice of adversarial training, for each mini-batch
            # we first train adv model and then embedding model
            for batch in train_loader:
                step += 1
                if step % 50 == 0:
                    print(f"Train step: {step} / {total_train_step}")
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

                # A tunable parameter, reduce num of training of adv model
                if step % args.adv_interval == 0:
                    adv_output = adv_dict['model'](embeddings_no_grad, hidden_states_no_grad)
                    # CrossEntropy loss target format (batch size, num_classes, d1, d2 ...)
                    adv_output = torch.transpose(adv_output, 1, 2)
                    adv_loss = adv_dict['loss_function'](adv_output, input_ids)
                    adv_loss.backward()
                    adv_dict['optimizer'].step()
                    adv_dict['optimizer'].zero_grad()
                    adv_train_loss += adv_loss.item()

                # After warmup epoch, start updating embedding model
                if epoch >= args.warmup_epochs:
                    # Train embedding
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
                    emebdding_loss = cls_loss - args.alpha * adv_loss
                    # emebdding_loss = cls_loss - 1.5 * adv_loss
                    emebdding_loss.backward()
                    embedding_dict['optimizer'].step()
                    embedding_dict['optimizer'].zero_grad()
                    embedding_train_adv_loss += adv_loss.item()
                    embedding_train_cls_loss += cls_loss.item()
                    embedding_dict['scheduler'].step()
                # train_progress_bar.update(1)

        elif args.adv_mode == 1:
            # In this training mode, we train adv and embedding separately
            # we first do mini-batch GD on the adv model, then on embedding model
            # Note this is not the common practice for adversarial training
            # but it shows big difference between train and val loss of adv
            # also this is less efficient because of the extra for loop
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

                # train_progress_bar.update(1)

            step = 0
            # train_progress_bar.refresh()
            # train_progress_bar.reset()

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
                embedding_loss = cls_loss - adv_loss
                embedding_loss.backward()
                embedding_dict['optimizer'].step()
                embedding_dict['optimizer'].zero_grad()
                embedding_train_adv_loss += adv_loss.item()
                embedding_train_cls_loss += cls_loss.item()

                # train_progress_bar.update(1)

        # Adapt to interval
        adv_train_loss /= (step / args.adv_interval)
        embedding_train_adv_loss /= step
        embedding_train_cls_loss /= step
        print(f"\nepoch {epoch + 1} average adv train loss: {adv_train_loss:.4f}")
        print(f"epoch {epoch + 1} average embedding train adv loss: {embedding_train_adv_loss:.4f}, average embedding train cls loss: {embedding_train_cls_loss:.4f}")

        # Record
        recorder['adv_train_loss'] = adv_train_loss
        recorder['embedding_train_adv_loss'] = embedding_train_adv_loss
        recorder['embedding_train_cls_loss'] = embedding_train_cls_loss


        if (epoch + 1) % args.val_interval == 0:
            step = 0
            adv_val_loss = 0
            embedding_val_cls_loss = 0

            adv_dict['model'].eval()
            embedding_dict['base_model'].eval()
            embedding_dict['classifier'].eval()

            # val_progress_bar.refresh()
            # val_progress_bar.reset()

            conf_mat = ConfusionMatrix(n_classes=2, device=device)

            with torch.no_grad():
                for batch in val_loader:
                    step += 1
                    if step % 20 == 0:
                        print(f"Val step: {step} / {total_val_step}")
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
                    conf_mat += torch.argmax(cls_output, dim=1), label
                    cls_loss = embedding_dict['loss_function'](cls_output, label)
                    embedding_val_cls_loss += cls_loss.item()

                    # val_progress_bar.update(1)
                    
                # Decode the last batch
                # output_ids = torch.argmax(adv_output, dim=1).detach().cpu()
                # original_seq = tokenizer.batch_decode(input_ids.cpu())
                # pred_seq = tokenizer.batch_decode(output_ids)
                # for ref, pred in zip(original_seq, pred_seq):
                #     print(f"Original input: {ref} \nAdv output: {pred}")

            adv_val_loss /= step
            embedding_val_cls_loss /= step
            tn, fn, fp, tp = conf_mat.value
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
            f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            print(f"\nepoch {epoch + 1} average adv val loss: {adv_val_loss:.4f}")
            print(f"epoch {epoch + 1} average embedding val cls loss: {embedding_val_cls_loss:.4f}, acc: {acc}, f1 score: {f1_score}, precision: {precision}")
            recorder['adv_val_loss'] = adv_val_loss
            recorder['embedding_val_cls_loss'] = embedding_val_cls_loss
            recorder['acc'] = acc
            recorder['f1_score'] = f1_score
            recorder['precision'] = precision

    # Save final model
    torch.save({'base_state_dict': embedding_dict['base_model'].state_dict(),'cls_state_dict': embedding_dict['classifier'].state_dict()}, f"{args.model_dir}adv_ft_{'_'.join(param_lst)}.pth")

    # train_progress_bar.close()
    # val_progress_bar.close()

    return recorder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--downsample", type=float, default=0.25)
    parser.add_argument("--adv_mode", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adv_learning_rate", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--adv_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--model_dir", type=str, default="model/")

    args = parser.parse_args()

    train_dataset, train_loader, val_dataset, val_loader = load_data("data/qqp_embedding_train.tsv", args.downsample, args.batch_size, args.batch_size)
    
    # Device
    device = torch.device("cuda:0")

    # Bert Model
    bert_config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = BertModel.from_pretrained("bert-base-uncased", config=bert_config)
    cls_model = BinaryClassificationHead(input_size=bert_config.hidden_size)
    embedding_optimizer = torch.optim.Adam([
        {"params": bert_model.parameters(), "lr": args.learning_rate},
        {"params": cls_model.parameters(), "lr": args.learning_rate},
        ])
    embedding_scheduler = torch.optim.lr_scheduler.LinearLR(embedding_optimizer, start_factor=1.0, end_factor=0.5, total_iters=2000)

    embedding_dict = {
        'base_model': bert_model,
        'classifier': cls_model,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': embedding_optimizer,
        'scheduler': embedding_scheduler
    }

    # Adv model
    adv_model = AdversarialDecoder(tgt_vocab_size=bert_config.vocab_size, device=device)
    adv_optimizer = torch.optim.Adam(adv_model.parameters(), args.adv_learning_rate)
    adv_dict = {
        'model': adv_model,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': adv_optimizer
    }

    param_lst = [args.downsample, args.adv_mode, args.num_epochs, args.warmup_epochs, args.learning_rate, args.adv_learning_rate, args.alpha, args.adv_interval]

    recorder = ResultRecorder(train_mode="adv_ft", params=param_lst)

    recorder = train_adv(train_loader, val_loader, adv_dict, embedding_dict, recorder, device, args)

    recorder.save(output_dir=args.output_dir)


