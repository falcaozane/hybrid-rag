"""
Sparse (BM25) retriever using rank_bm25.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi


def retrieve(
    query: str,
    bm25_index: BM25Okapi,
    chunks: list[str],
    chunk_pages: list[int],
    k: int = 5,
) -> list[tuple[str, float, int]]:
    """
    Score all chunks against *query* using BM25, return top-k.

    Lowercase tokenisation must match what was used when building the index.
    Zero-score results are included — RRF fusion decides final ranking.

    Returns list of (chunk_text, bm25_score, page_num), sorted by score desc.
    """
    tokenized_query = query.lower().split()
    scores: np.ndarray = bm25_index.get_scores(tokenized_query)

    actual_k = min(k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    results: list[tuple[str, float, int]] = []
    for idx in top_indices:
        results.append((chunks[idx], float(scores[idx]), chunk_pages[idx]))

    return results