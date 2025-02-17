import torch
import json
import os
import sys
from transformers import BertTokenizer
from models.bert_regression import BertForRegression

def evaluate_model(text_input):
    # Load the finetuned model
    model_path = os.path.join("models", "finetuned", "bert_regression", "checkpoint-500")
    model = BertForRegression.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare input
    encoding = tokenizer(
        text_input,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        predicted_calories = outputs.squeeze().item() * 1000  # Scale back up from normalized value
    
    return predicted_calories

def evaluate_test_set():
    # Load test data
    eval_file = os.path.join("data", "eval", "usda_1146.json")
    with open(eval_file, 'r') as f:
        test_data = json.load(f)
    
    # Calculate metrics
    total_error = 0
    total_samples = len(test_data)
    
    for food_desc, actual_calories in test_data:
        predicted_calories = evaluate_model(food_desc)
        error = abs(predicted_calories - actual_calories)
        total_error += error
        
        print(f"Food: {food_desc}")
        print(f"Actual calories: {actual_calories:.1f}")
        print(f"Predicted calories: {predicted_calories:.1f}")
        print(f"Error: {error:.1f}\n")
    
    mae = total_error / total_samples
    print(f"Mean Absolute Error: {mae:.1f} calories")

if __name__ == "__main__":
    # Test single prediction
    sample_text = "grilled chicken breast with rice"
    calories = evaluate_model(sample_text)
    print(f"Predicted calories for '{sample_text}': {calories:.1f}")
    
    # Evaluate on test set
    print("\nEvaluating test set:")
    evaluate_test_set()