#!/usr/bin/env python3

import argparse
import json
import string
import sys
from nltk.stem import PorterStemmer

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

    args = parser.parse_args()
    translation_table = str.maketrans("", "", string.punctuation)

    match args.command:
        case "search":
            search_movies(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
