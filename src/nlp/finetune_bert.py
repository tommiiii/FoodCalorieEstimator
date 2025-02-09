import json
import torch
import os
import glob
import re
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments

class FoodDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.examples = []
        for item in data:
            text = item
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
            encoding['labels'] = encoding['input_ids'].copy()
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

def find_largest_numbered_usda_file(directory):
    pattern = os.path.join(directory, "usda_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No USDA files found in {directory}")
    
    numbers = []
    for f in files:
        match = re.search(r'usda_(\d+)\.json', f)
        if match:
            numbers.append((int(match.group(1)), f))
    
    if not numbers:
        raise ValueError(f"No valid USDA file numbers found in {directory}")
    
    # Return the file path with the largest number
    return max(numbers)[1]

def main():
    # Find the USDA files with largest numbers in train and eval directories
    train_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")
    eval_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval")
    
    data_file_train = find_largest_numbered_usda_file(train_dir)
    data_file_eval = find_largest_numbered_usda_file(eval_dir)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    train_dataset = FoodDataset(data_file_train, tokenizer)
    eval_dataset = FoodDataset(data_file_eval, tokenizer)
    
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
    main()