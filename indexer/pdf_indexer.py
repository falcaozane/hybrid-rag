"""
PDF Indexer — PyMuPDF extraction, token-based chunking,
FAISS (cosine via IndexFlatIP) and BM25 index builders.
"""

from __future__ import annotations

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Singleton model — loaded once per Python process to avoid paying the
# ~2 s / ~90 MB load cost on every query or rerun.
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """
    Extract text from a PDF supplied as raw bytes.

    Returns a list of (page_num, text) tuples (1-indexed page numbers).
    Pages with no extractable text are silently skipped.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            pages.append((page_num + 1, text))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = 200,
    overlap: int = 50,
) -> tuple[list[str], list[int]]:
    """
    Flatten all pages into a token stream, then produce overlapping chunks.

    - chunk_size: target number of whitespace-split tokens per chunk
    - overlap:    number of tokens shared between consecutive chunks
    - Each chunk is attributed to the page of its *first* token.

    Returns (chunks, chunk_pages).
    """
    all_tokens: list[str] = []
    token_pages: list[int] = []

    for page_num, text in pages:
        tokens = text.split()
        all_tokens.extend(tokens)
        token_pages.extend([page_num] * len(tokens))

    if not all_tokens:
        return [], []

    step = chunk_size - overlap  # stride = 250 by default
    chunks: list[str] = []
    chunk_pages: list[int] = []
    i = 0

    while i < len(all_tokens):
        window_tokens = all_tokens[i : i + chunk_size]
        window_pages = token_pages[i : i + chunk_size]

        chunk_str = " ".join(window_tokens)
        dominant_page = window_pages[0]

        chunks.append(chunk_str)
        chunk_pages.append(dominant_page)

        # Final partial window — stop after capturing it
        if len(window_tokens) < chunk_size:
            break

        i += step

    return chunks, chunk_pages


# ---------------------------------------------------------------------------
# ChromaDB index
# ---------------------------------------------------------------------------

def build_chroma_index(chunks: list[str], chunk_pages: list[int]):
    """
    Build an in-memory ChromaDB collection with the chunks and their pages.
    """
    client = chromadb.EphemeralClient()
    
    # We use the same model "all-MiniLM-L6-v2" as the embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.create_collection(
        name="rag_docs",
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    collection.add(
        documents=chunks,
        metadatas=[{"page": p} for p in chunk_pages],
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    
    return collection


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """
    Lowercase-tokenise every chunk and return a BM25Okapi index.
    Lowercasing is critical — BM25 is case-sensitive by default.
    """
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)