import os
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import faiss
import torch

# Load model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "finetuned")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained(model_path)

def compute_embedding(text: str) -> np.ndarray:
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get the model outputs with no gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the hidden states
    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
    
    # Create attention mask
    attention_mask = inputs['attention_mask']
    
    # Perform mean pooling
    # First, expand attention mask to 3D
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    
    # Sum the hidden states using the attention mask
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    
    # Sum the attention mask to get the actual length for each sequence
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Calculate mean by dividing sum_embeddings by sum_mask
    mean_embedding = (sum_embeddings / sum_mask).numpy()
    
    # Normalize the embedding for cosine similarity
    norm_embedding = mean_embedding / np.linalg.norm(mean_embedding, axis=1, keepdims=True)
    return norm_embedding

# Example texts to index
texts = [
    "This is the first example.",
    "Here is another example sentence.",
    "This is a different text altogether."
]

# Compute embeddings and stack them into an array
embeddings = np.vstack([compute_embedding(text) for text in texts])

# Create a FAISS index for cosine similarity
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

## TESTS ##
# To perform a search, compute the embedding for a query and search the index:
query = "here."
query_embedding = compute_embedding(query)

# Searching for top 2 similar items
k = 2
distances, indices = index.search(query_embedding, k)

print("Similar texts:")
for i, idx in enumerate(indices[0]):
    print(f"{texts[idx]} - Similarity: {distances[0][i]}")