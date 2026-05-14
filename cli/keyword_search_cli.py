#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command
)

from lib.search_utils import BM25_B, BM25_K1

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=str, help="document ID")
    tf_parser.add_argument("term", type=str, help="term to check frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="term to check IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=str, help="document ID")
    tfidf_parser.add_argument("term", type=str, help="term to check TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25_idf_parser.add_argument("term", type=str, help="term to check BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a term in a document")
    bm25_tf_parser.add_argument("doc_id", type=str, help="document ID")
    bm25_tf_parser.add_argument("term", type=str, help="term to check BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?',             default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building the inverted index...")
            build_command()
            print("Index built successfully.")
        case "search":
            print(f"Searching for: '{args.query}'")
            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. ({result['id']}) {result['title']}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}' with k1={args.k1}: {bm25tf:.2f}")
        case "bm25search":
            print(f"Searching for: '{args.query}' using BM25 scoring")
            results = bm25search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
