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
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_or_create_embeddings() first.")
        
        query_embedding = self.generate_embedding(query)
        similarities = np.array([cosine_similarity(query_embedding, emb) for emb in self.embeddings])
        # Create a list of (similarity, document) tuples
        similarity_doc_pairs = [(similarity, self.document_map[doc['id']]) for similarity, doc in zip(similarities, self.documents)]
        # Sort by similarity in descending order
        similarity_doc_pairs.sort(key=lambda x: x[0], reverse=True)
        # Get the top results based on the limit as a list of dictionaries containing score, title, and description 
        results = [{"score": similarity, **doc} for similarity, doc in similarity_doc_pairs[:limit]]

        return results
def search_preprocess(query, limit):
    ss = SemanticSearch()
    # Load movies and load or create embeddings
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    documents = data["movies"]
    ss.load_or_create_embeddings(documents)
    results = ss.search(query, limit)
    # Print the results in this format:
    # 1. <title> (score: <score>)
    #   <description>
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}")

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

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def chunk(text, chunk_size, overlap):
    total_length = len(text)
    words = text.split()

    chunks = []
    start = 0
    while start < len(words) - overlap:
        chunk = words[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

        if chunk_size <= overlap:
            break

    print(f"\nChunking {total_length} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
