#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_query_text, search_preprocess, verify_embeddings, verify_model, embed_text, verify_embeddings, chunk, semantic_chunk, embed_chunks

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed a text string")
    embed_text_parser.add_argument("text", help="The text to embed")

    verify_embed_parser = subparsers.add_parser("verify_embeddings", help="Verify embedding generation and caching")

    query_embed_parser = subparsers.add_parser("embed_query", help="Embed a query text string")
    query_embed_parser.add_argument("query", help="The query text to embed")

    search_parser = subparsers.add_parser("search", help="Search for similar documents")
    search_parser.add_argument("query", help="The query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Test text chunking")
    chunk_parser.add_argument("text", help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=4, help="The size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="The number of overlapping words between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Test semantic text chunking")
    semantic_chunk_parser.add_argument("text", help="The text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="The maximum size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="The number of overlapping words between chunks")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Test embedding generation for text chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_preprocess(args.query, args.limit)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()