#!/usr/bin/env python3

import argparse
import json
import math
import string
import sys
import os
import pickle
from nltk.stem import PorterStemmer
from collections import Counter

class InvertedIndex:
    def __init__(self, index=None, docmap=None, term_frequencies=None):
        self.index= index if index is not None else {}
        self.docmap = docmap if docmap is not None else {}
        self.term_frequencies = term_frequencies if term_frequencies is not None else {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id] = Counter(tokens)

    def get_documents(self, term):
        term = term.lower()

        ids_set = self.index.get(term, set())
        return sorted(list(ids_set))

    def build(self, movies):
        for movie in movies:
            # 1. Get the ID (assuming the key is 'id')
            doc_id = movie["id"]
            
            # 2. Add the full movie object to the docmap
            self.docmap[doc_id] = movie
            
            # 3. Concatenate title and description
            # Using an f-string as suggested in the prompt
            full_text = f"{movie['title']} {movie['description']}"
            
            # 4. Use your helper to index the words
            self.__add_document(doc_id, full_text)

    def save(self):
        # 1. Create the 'cache' directory if it doesn't exist
        if not os.path.exists("cache"):
            os.makedirs("cache")

        # 2. Save the index to cache/index.pkl
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        # 3. Save the docmap to cache/docmap.pkl
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        # Save term_frequencies to cache/term_frequencies.pkl
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        # Check if both required files exist in the cache
        if not os.path.exists("cache/index.pkl") or not os.path.exists("cache/docmap.pkl") or not os.path.exists("cache/term_frequencies.pkl"):
            raise FileNotFoundError("Index, Docmap, or Term Frequency file not found in cache")

        # Load the inverted index dictionary
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        # Load the document mapping dictionary
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        # Load the term_frequency dictionary
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id, term):
        # Tokenize the term
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("Please provide a single term for term frequency calculation.")
        
        token = tokens[0]
        doc_counts = self.term_frequencies.get(doc_id, {})
        return doc_counts.get(token, 0)
    
    def get_bm25_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        doc_freq = len(self.get_documents(term))
        return math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

PUNC_TABLE = str.maketrans("", "", string.punctuation)

stemmer = PorterStemmer()

def get_movies(path="data/movies.json"):
    try:
        with open('data/movies.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: data/movies.json not found.")
        sys.exit(1)

def get_stopwords(path="data/stopwords.txt"):
    try:
        with open(path, "r") as f:
            # splitlines() gives us a list, set() makes lookups instant
            return set(f.read().splitlines())
    except FileNotFoundError:
        return set()

def tokenize(text, stopwords=None):
    # 1. Clean and split
    clean_text = text.translate(PUNC_TABLE).lower()
    words = clean_text.split()
    
    # 2. Filter out stop words if provided
    if stopwords:
        return [stemmer.stem(w) for w in words if w not in stopwords]
   
    return [stemmer.stem(w) for w in words]

def search_movies(query):
    # Initialize and load the index
    idx = InvertedIndex()
    try: 
        idx.load()
    except FileNotFoundError:
        print("Index not found.  Please run the 'build' command first.")
        sys.exit(1)

    # Tokenize the user's query
    #movies = get_movies()
    stop_words = get_stopwords()

    query_tokens = tokenize(query, stop_words)
    results_ids = []

    for token in query_tokens:
        ids = idx.get_documents(token)
        for doc_id in ids:
            if doc_id not in results_ids:
                results_ids.append(doc_id)

            # Stop once we have 5 unique results
            if len(results_ids) >= 5:
                break
        if len(results_ids) >= 5:
            break

    # Print top 5 results
    for doc_id in results_ids:
        movie = idx.docmap.get(doc_id)
        if movie:
            print(f"{movie['title']} ({doc_id})")

def term_freq(doc_id, term):
    # Initialize and load the index
    idx = InvertedIndex()
    try: 
        idx.load()
    except FileNotFoundError:
        print("Index not found.  Please run the 'build' command first.")
        sys.exit(1)

    tf = idx.get_tf(int(doc_id), term)
    print(tf)

def idf(term):
    # Initialize and load the index
    idx = InvertedIndex()
    try: 
        idx.load()
    except FileNotFoundError:
        print("Index not found.  Please run the 'build' command first.")
        sys.exit(1)

    # Tokenize / Stem the term
    tokens = tokenize(term)
    search_term = tokens[0] if tokens else term

    # Get total number of documents
    total_docs = len(idx.docmap)

    # Get number of documents containing the term
    doc_freq = len(idx.get_documents(search_term))

    idf_value = math.log((total_docs + 1) / (doc_freq + 1)) 
    # Adding 1 to avoid division by zero
    print(f"Inverse document frequency of '{term}': {idf_value:.2f}")

def tfidf(doc_id, term):
    # Initialize and load the index
    idx = InvertedIndex()
    try: 
        idx.load()
    except FileNotFoundError:
        print("Index not found.  Please run the 'build' command first.")
        sys.exit(1)

    # Tokenize / Stem the term
    tokens = tokenize(term)
    search_term = tokens[0] if tokens else term

    # Get TF
    tf = idx.get_tf(int(doc_id), search_term)

    # Get IDF
    total_docs = len(idx.docmap)
    doc_freq = len(idx.get_documents(search_term))
    idf_value = math.log((total_docs + 1) / (doc_freq + 1)) 

    tfidf_score = tf * idf_value
    print(f"TF-IDF score for term '{term}' in document '{doc_id}': {tfidf_score:.2f}")    

def bm25_idf(term):
    # Initialize and load the index
    idx = InvertedIndex()
    try: 
        idx.load()
    except FileNotFoundError:
        print("Index not found.  Please run the 'build' command first.")
        sys.exit(1)

    # Tokenize / Stem the term
    tokens = tokenize(term)
    search_term = tokens[0] if tokens else term

    bm25_idf_value = idx.get_bm25_idf(search_term)
    print(f"BM25 IDF score for term '{term}': {bm25_idf_value:.2f}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=str, help="document ID")
    tf_parser.add_argument("term", type=str, help="term to check frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="term to check IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=str, help="document ID")
    tfidf_parser.add_argument("term", type=str, help="term to check TF-IDF score for")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25idf_parser.add_argument("term", type=str, help="term to check BM25 IDF score for")

    args = parser.parse_args()
    translation_table = str.maketrans("", "", string.punctuation)

    match args.command:
        case "search":
            search_movies(args.query)
        case "build":
            # Load the raw data from the JSON file
            movies_data = get_movies()
            # Access the list of movie dictionaries
            movie_list = movies_data.get("movies", [])
            # Instantiate with empty dicts for index and docmap
            idx = InvertedIndex()
            # Build and save
            idx.build(movie_list)
            idx.save()
        case "tf":
            term_freq(args.doc_id, args.term)
        case "idf":
            idf(args.term)
        case "tfidf":
            tfidf(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf(args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
