# language: python
import argparse
import numpy as np
import faiss
from transformers import BertForMaskedLM, BertTokenizer
from .extract_embeddings import encode_text_static, fetch_descriptions

def search_faiss_index():
    parser = argparse.ArgumentParser(
        description="Search a FAISS index using a string query and return matching food descriptions."
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to the Faiss index file (e.g., my_index.index)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory used when indexing"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query string to search for"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve"
    )
    args = parser.parse_args()

    # Load the FAISS index.
    index = faiss.read_index(args.index)

    # Load the model to extract the static embedding matrix.
    model = BertForMaskedLM.from_pretrained(args.model_path)
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # Load the tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Encode the query string as a static embedding.
    query_embedding = encode_text_static(args.query, tokenizer, embedding_matrix)
    query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {index.d}"
        )

    # Search the index.
    distances, indices = index.search(query_embedding, args.k)

    # Load the original food descriptions 
    # (assumed to be in the same order they were indexed).
    food_descriptions = fetch_descriptions()

    print("Nearest Neighbors:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        try:
            description = food_descriptions[idx]
        except IndexError:
            description = "Unknown description (index out of bounds)"
        print(f"Neighbor {i + 1}:")
        print(f"  Description: {description}")
        print(f"  Index: {idx}")
        print(f"  Distance: {dist}\n")

if __name__ == "__main__":
    search_faiss_index()