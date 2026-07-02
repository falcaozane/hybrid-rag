"""
PDF Indexer — PyMuPDF extraction, token-based chunking,
FAISS (cosine via IndexFlatIP) and BM25 index builders.
"""

from __future__ import annotations

import numpy as np
import faiss
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
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(chunks: list[str]) -> faiss.IndexFlatIP:
    """
    Encode chunks with all-MiniLM-L6-v2 (L2-normalised) and build an
    IndexFlatIP.  Inner product on unit vectors == cosine similarity.
    """
    model = _get_model()
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


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