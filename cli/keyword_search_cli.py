#!/usr/bin/env python3

import argparse
import json
import string
import sys
import os
import pickle
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self, index=None, docmap=None):
        self.index= index if index is not None else {}
        self.docmap = docmap if docmap is not None else {}

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

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

    def load(self):
        


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
    movies = get_movies()
    stop_words = get_stopwords()

    query_tokens = tokenize(query, stop_words)
    results = []

    for movie in movies.get("movies", []):
        title_tokens = tokenize(movie["title"], stop_words)
        if any(q_t in t_t for q_t in query_tokens for t_t in title_tokens):
            results.append(movie)

    # Print top 5 results
    for i, movie in enumerate(results[:5], 1):
        print(f"{i}. {movie["title"]}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
