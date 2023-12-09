import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class QQPDataSet(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        question1 = str(self.data.question1[index])
        question2 = str(self.data.question2[index])
        label = int(self.data.label[index])

        inputs = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len


class BERTClass(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClass, self).__init__()
        self.bert = bert_model
        self.linear = nn.Linear(768, num_classes)

    def forward(self, ids, mask):
        output = self.bert(ids, attention_mask=mask)
        mean_pooled = output[0].mean(1)
        x = self.linear(mean_pooled)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

train_df = pd.read_parquet('./qqp/train/0000.parquet')
validation_df = pd.read_parquet('./qqp/validation/0000.parquet')

train_sample = int(len(train_df) / (len(train_df) + len(validation_df)) * 60000)
validation_sample = int(len(validation_df) / (len(train_df) + len(validation_df)) * 60000)

train_df = train_df.sample(n=train_sample).reset_index(drop=True)
validation_df = validation_df.sample(n=validation_sample).reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_set = QQPDataSet(train_df, tokenizer)
validation_set = QQPDataSet(validation_df, tokenizer)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=128)

learning_rates = [5e-5, 1e-5, 1e-6]
num_epochs = 5

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")

    model = BERTClass(BertModel.from_pretrained('bert-base-uncased'), num_classes=2).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_index, data in enumerate(tqdm(train_loader), 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % 500 == 0:
                print(f"Batch {batch_index}/{len(train_loader)} - Loss: {loss.item()}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss}")

        model.eval()
        total_eval_loss = 0
        true_labels = []
        predictions = []
        with torch.no_grad():
            for data in validation_loader:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.long)

                outputs = model(ids, mask)
                loss = loss_function(outputs, labels)
                total_eval_loss += loss.item()

                true_labels.extend(labels.cpu().numpy())
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        avg_val_loss = total_eval_loss / len(validation_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}, F1 Score: {val_f1}")
