# language: python
import os
import argparse
import numpy as np
import faiss
import json
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

def create_index(model_path):
    """
    Create a Faiss index from the static word embeddings of the model.
    """
    # Load the finetuned masked LM model.
    model = BertForMaskedLM.from_pretrained(model_path)
    # Extract the embedding matrix from the word embeddings.
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    
    # Create a Faiss index for the vocabulary embeddings.
    d = embedding_matrix.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embedding_matrix)
    
    return index, embedding_matrix

def fetch_descriptions() -> list:
    """
    Load USDA datasets from train and eval directories and return a list of food descriptions.
    """
    usda_file = os.path.join("data", "train", "usda_4582.json")
    with open(usda_file, 'r') as f:
        usda_data = json.load(f)
    
    food_descriptions = [food_desc for food_desc, _ in usda_data]
    
    usda_file = os.path.join("data", "eval", "usda_1146.json")
    with open(usda_file, 'r') as f:
        usda_data = json.load(f)
    
    food_descriptions += [food_desc for food_desc, _ in usda_data]
    
    return food_descriptions

def parse_arguments():
    """Parse command line arguments to create configuration."""
    parser = argparse.ArgumentParser(
        description="Extract static embeddings from a MaskedLM BERT model and save the Faiss index.",
        epilog="Example: uv run extract_embeddings --model-path models/finetuned/maskedlm/checkpoint-500 --output my_index.index"
    )
    parser.add_argument("--model-path", type=str, help="Path to the model directory", required=True)
    parser.add_argument("--output", type=str, help="File path where the Faiss index will be saved", default="faiss.index")
    args = parser.parse_args()
    return args.model_path, args.output

def encode_text_static(text: str, tokenizer: BertTokenizer, embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Encode a text string as a static embedding by tokenizing the text, looking up each token in
    the embedding matrix, and mean-pooling them.
    """
    # Tokenize text to obtain a list of token ids.
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # Get the static embeddings for each token.
    token_embeddings = embedding_matrix[tokens]  # shape: (seq_len, emb_dim)
    
    # If no tokens were found, return a zero vector.
    if len(token_embeddings) == 0:
        return np.zeros((embedding_matrix.shape[1],), dtype="float32")
    
    # Return the mean of the token embeddings.
    return token_embeddings.mean(axis=0)

def extract_model_embeddings():
    """
    Extract the static vocabulary embeddings and create a Faiss index.
    """
    model_path, output = parse_arguments()
    index, embedding_matrix = create_index(model_path)
    print("Static Index created successfully with number of vectors:", index.ntotal)
    print("Index dimensions:", index.d)
    
    # Save the Faiss index.
    faiss.write_index(index, output)
    print(f"Faiss index saved successfully to: {output}")

def dataset_faiss_embeddings_index():
    """
    Encode food descriptions using static token embeddings (mean pooling) and build a Faiss index.
    """
    model_path, output = parse_arguments()
    
    # Load the model once to extract the static embedding matrix.
    model = BertForMaskedLM.from_pretrained(model_path)
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    
    # Load the tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    food_descriptions = fetch_descriptions()
    description_embeddings = []
    
    for description in tqdm(food_descriptions, desc="Processing descriptions"):
        emb = encode_text_static(description, tokenizer, embedding_matrix)
        description_embeddings.append(emb)
    
    description_embeddings = np.vstack(description_embeddings)
    
    # Build a Faiss index for the description embeddings.
    index = faiss.IndexFlatL2(description_embeddings.shape[1])
    index.add(description_embeddings)
    print("Embeddings indexed successfully with number of vectors:", index.ntotal)
    
    faiss.write_index(index, output)
    print(f"Faiss index saved successfully to: {output}")

def main():
    extract_model_embeddings()
    dataset_faiss_embeddings_index()

if __name__ == "__main__":
    main()