from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Training on CPU")

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

train_df = pd.read_parquet('./qqp/train/0000.parquet')
validation_df = pd.read_parquet('./qqp/validation/0000.parquet')
test_df = pd.read_parquet('./qqp/test/0000.parquet')

train_ds = Dataset.from_pandas(train_df)
validation_ds = Dataset.from_pandas(validation_df)
test_ds = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': train_ds,
    'validation': validation_ds,
    'test': test_ds
})

def encode(examples):
    return tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length')

encoded_dataset = dataset.map(encode, batched=True)

train_test_split = encoded_dataset['train'].train_test_split(train_size=0.6)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

aux_dataset = test_dataset.shuffle(seed=42).select(range(0, int(len(test_dataset) * 0.5)))
target_dataset = test_dataset.shuffle(seed=42).select(range(int(len(test_dataset) * 0.5), len(test_dataset)))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=36,
    per_device_eval_batch_size=36,
    warmup_steps=250,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer
)

torch.cuda.empty_cache()
trainer.train()

torch.cuda.empty_cache()
results = trainer.evaluate()
print(results)

model.save_pretrained('./finetuned_bert_qqp')
tokenizer.save_pretrained('./finetuned_bert_qqp')
