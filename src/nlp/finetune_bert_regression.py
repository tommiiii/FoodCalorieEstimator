import json
import math
import torch
import os
import glob
import re
import dotenv
from torch.utils.data import Dataset
from transformers import BertTokenizer, Trainer, TrainingArguments
from models.bert_regression import BertForRegression

#IMPORTANT: this finetunes the masked language model. Embeddings don't need to be extracted to be used with faiss, so this model has to be used as-is after.

class FoodDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.examples = []
        for item in data:
            text = item[0]
            calories = item[1]
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
            encoding['labels'] = torch.tensor(calories, dtype=torch.float)
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

def main():
    # Find the USDA files with largest numbers in train and eval directories
    train_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train", os.getenv("TRAIN_FILE", "usda"))
    eval_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval", os.getenv("EVAL_FILE", "usda"))
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model = BertForRegression.from_pretrained("bert-base-uncased")
    
    train_dataset = FoodDataset(train_dir, tokenizer)
    eval_dataset = FoodDataset(eval_dir, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "..", "..", "models", "finetuned"),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    
if __name__ == "__main__":
    dotenv.load_dotenv()
    main()