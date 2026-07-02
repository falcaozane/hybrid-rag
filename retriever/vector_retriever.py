"""
Dense (FAISS) retriever using the shared all-MiniLM-L6-v2 singleton.
"""

from __future__ import annotations

import numpy as np
import faiss

# Reuse the singleton loaded by the indexer — avoids a second 90 MB load.
from indexer.pdf_indexer import _get_model


def retrieve(
    query: str,
    faiss_index: faiss.IndexFlatIP,
    chunks: list[str],
    chunk_pages: list[int],
    k: int = 5,
) -> list[tuple[str, float, int]]:
    """
    Encode *query*, search the FAISS index, return top-k results.

    Returns list of (chunk_text, cosine_score, page_num), sorted by score desc.
    """
    model = _get_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.array(query_embedding, dtype="float32")

    actual_k = min(k, len(chunks))
    scores, indices = faiss_index.search(query_embedding, actual_k)

    results: list[tuple[str, float, int]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS padding sentinel — fewer results than k
            continue
        results.append((chunks[idx], float(score), chunk_pages[idx]))

    return results