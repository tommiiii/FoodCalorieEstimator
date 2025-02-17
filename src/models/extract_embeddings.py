# language: python
import os
import argparse
import numpy as np
import faiss
import json
import torch
from transformers import BertForMaskedLM, BertTokenizer

def create_index(model_path):
    # Load the finetuned masked LM model.
    model = BertForMaskedLM.from_pretrained(model_path)

    # Extract the embedding matrix.
    # For BERT-based models, the word embeddings are stored in model.bert.embeddings.word_embeddings.weight
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # Create a Faiss index.
    d = embedding_matrix.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embedding_matrix)
    
    return index, embedding_matrix

def fetch_descriptions() -> list:
    # Load the USDA dataset
    usda_file = os.path.join("data", "train", "usda_4582.json")
    with open(usda_file, 'r') as f:
        usda_data = json.load(f)
    
    # Extract the food descriptions
    food_descriptions = [food_desc for food_desc, _ in usda_data]
    
    usda_file = os.path.join("data", "eval", "usda_1146.json")
    with open(usda_file, 'r') as f:
        usda_data = json.load(f)
    
    food_descriptions += [food_desc for food_desc, _ in usda_data]

    return food_descriptions

def parse_arguments():
    """Parse command line arguments to create configuration."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a MaskedLM BERT model and save the Faiss index.",
        epilog="Example: uv run extract_embeddings --model-path models/finetuned/maskedlm/checkpoint-500 --output my_index.index"
    )
    parser.add_argument("--model-path", type=str, help="Path to the model directory", required=True)
    parser.add_argument("--output", type=str, help="File path where the Faiss index will be saved", default="faiss.index")
    args = parser.parse_args()
    return args.model_path, args.output

def extract_model_embeddings():
    model_path, output_index = parse_arguments()
    index, embedding_matrix = create_index(model_path)
    print("Index created successfully with number of vectors:", index.ntotal)
    print("Index dimensions:", index.d)
    
    # Save the Faiss index to the specified file.
    faiss.write_index(index, output_index)
    print(f"Faiss index saved successfully to: {output_index}")
    
def dataset_faiss_embeddings_index():
    model_path, output_index = parse_arguments()
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    food_descriptions = fetch_descriptions()
    description_embeddings = []

    for description in food_descriptions:
        inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            token_embeddings = outputs.hidden_states[-1]            # Mean pooling
            attention_mask = inputs['attention_mask']
            # Expand attention mask to same dims as embeddings
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # Sum up embeddings where attention mask is 1
            sum_embeddings = torch.sum(token_embeddings * mask, 1)
            # Sum up mask values
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            # Average
            embedding = (sum_embeddings / sum_mask).cpu().numpy()
            description_embeddings.append(embedding)

    description_embeddings = np.vstack(description_embeddings)
    index = faiss.IndexFlatL2(description_embeddings.shape[1])
    index.add(description_embeddings)
    print("Embeddings indexed successfully with number of vectors:", index.ntotal)

    faiss.write_index(index, output_index)
    print(f"Faiss index saved successfully to: {output_index}")
    
    
def test_dataset_faiss_embeddings_index_with_sentence(
    sentence: str = "This is a test sentence for FAISS indexing.",
    model_path: str = "bert-base-uncased",
    output_index: str = "test.index"
):
    """
    Test the dataset_faiss_embeddings_index function with a single sentence.

    This function temporarily monkeypatches parse_arguments() and fetch_descriptions()
    so that instead of reading command-line arguments and files, it returns fixed values,
    allowing you to test the embedding extraction with the provided sentence.
    """
    global parse_arguments, fetch_descriptions

    # Save the original functions
    original_parse_arguments = parse_arguments
    original_fetch_descriptions = fetch_descriptions

    # Override to return test values
    def dummy_parse_arguments():
        return model_path, output_index

    def dummy_fetch_descriptions() -> list:
        return [sentence]

    parse_arguments = dummy_parse_arguments
    fetch_descriptions = dummy_fetch_descriptions

    try:
        dataset_faiss_embeddings_index()
    finally:
        # Restore the original functions
        parse_arguments = original_parse_arguments
        fetch_descriptions = original_fetch_descriptions

def main():
    extract_model_embeddings()
    dataset_faiss_embeddings_index()

if __name__ == "__main__":
    main()