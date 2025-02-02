import json
import torch
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class FoodDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.examples = []
        for item in data:
            text = item["description"]
            label = item.get("label", 0)  # Update as needed for your task
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length)
            encoding["label"] = label
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

def main():
    # Update this path to your JSON data file
    data_file = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "data", "processed", "usda_food_foundation_data_filtered.json"
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Set the number of labels according to your classification task:
    NUMBER_OF_LABELS = 2  # Example: binary classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUMBER_OF_LABELS)
    
    dataset = FoodDataset(data_file, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results",
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
        train_dataset=dataset,
        eval_dataset=dataset  # Replace with a separate eval dataset if available.
    )
    
    trainer.train()

if __name__ == "__main__":
    main()