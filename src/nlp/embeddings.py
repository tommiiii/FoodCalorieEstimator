from transformers import BertTokenizer, BertModel
import numpy as np
import faiss

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def compute_embedding(text: str) -> np.ndarray:
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Get the model outputs
    outputs = model(**inputs)
    # Use [CLS] token representation for the embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    # Normalize the embedding (for cosine similarity)
    norm_embedding = cls_embedding / np.linalg.norm(cls_embedding, axis=1, keepdims=True)
    return norm_embedding

# Example texts to index
texts = [
    "This is the first example.",
    "Here is another example sentence.",
    "This is a different text altogether."
]

# Compute embeddings and stack them into an array
embeddings = np.vstack([compute_embedding(text) for text in texts])

# Create a FAISS index for cosine similarity.
# Since the embeddings are normalized, using IndexFlatIP will compute cosine similarity.
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)


## TESTS ##
# To perform a search, compute the embedding for a query and search the index:
query = "A sample query text."
query_embedding = compute_embedding(query)

# Searching for top 2 similar items
k = 2
distances, indices = index.search(query_embedding, k)

print("Similar texts:")
for i, idx in enumerate(indices[0]):
    print(f"{texts[idx]} - Similarity: {distances[0][i]}")