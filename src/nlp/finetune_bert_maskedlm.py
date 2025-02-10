import json
import torch
import os

import dotenv
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments

# IMPORTANT: this finetunes the masked language model. This is only to extract embeddings from this model.

class FoodDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.examples = []
        for item in data:
            text = item[0]
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
            encoding['labels'] = encoding['input_ids'].copy()
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

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    train_dataset = FoodDataset(train_dir, tokenizer)
    eval_dataset = FoodDataset(eval_dir, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), "..", "..", "models", "finetuned"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
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
