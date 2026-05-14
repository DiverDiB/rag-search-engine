import json
import os
import numpy as np

from collections import defaultdict

from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, DATA_PATH

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict(list)

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Input text cannot be empty or whitespace.")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        strings_to_embed = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(strings_to_embed, show_progress_bar=True)

        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:

            self.document_map[doc['id']] = doc

        # Build path using CACHE_DIR from search_utils
        cache_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(cache_path):
            print("Loading cached embeddings...")
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return embedding

def verify_embeddings():
    ss = SemanticSearch()
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    documents = data["movies"]

    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")