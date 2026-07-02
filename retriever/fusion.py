"""
Reciprocal Rank Fusion (RRF) — pure Python, no external dependencies.

Formula:  score(d) = Σ  1 / (rrf_k + rank(d))
where rrf_k = 60 (standard constant from the original RRF paper).

NOTE: rrf_k=60 is the RRF *smoothing constant*, NOT the retrieval top-K.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float, int]],
    bm25_results: list[tuple[str, float, int]],
    rrf_k: int = 60,
) -> list[tuple[str, float, float, float, int, str]]:
    """
    Merge and re-rank two result lists using RRF.

    Each input list contains (chunk_text, score, page_num) tuples already
    sorted by descending score (rank 1 = best).

    Returns a list of:
        (chunk_text, rrf_score, vector_score, bm25_score, page_num, found_by)
    sorted by rrf_score descending.

    found_by ∈ {"Both", "Vector", "BM25"}
    """
    # Build lookup maps keyed by chunk text
    vector_map: dict[str, tuple[float, int]] = {
        chunk: (score, page) for chunk, score, page in vector_results
    }
    bm25_map: dict[str, tuple[float, int]] = {
        chunk: (score, page) for chunk, score, page in bm25_results
    }

    # 1-indexed rank maps (rank 1 = highest score)
    vector_ranks: dict[str, int] = {
        chunk: rank + 1 for rank, (chunk, _, _) in enumerate(vector_results)
    }
    bm25_ranks: dict[str, int] = {
        chunk: rank + 1 for rank, (chunk, _, _) in enumerate(bm25_results)
    }

    # Union of all unique chunks, preserving encounter order
    all_chunks: list[str] = list(
        dict.fromkeys(
            [c for c, _, _ in vector_results] + [c for c, _, _ in bm25_results]
        )
    )

    fused: list[tuple[str, float, float, float, int, str]] = []

    for chunk in all_chunks:
        rrf_score = 0.0
        if chunk in vector_ranks:
            rrf_score += 1.0 / (rrf_k + vector_ranks[chunk])
        if chunk in bm25_ranks:
            rrf_score += 1.0 / (rrf_k + bm25_ranks[chunk])

        v_score = vector_map[chunk][0] if chunk in vector_map else 0.0
        b_score = bm25_map[chunk][0] if chunk in bm25_map else 0.0

        # Resolve page number: prefer vector source, fall back to BM25
        page_num = (vector_map.get(chunk) or bm25_map.get(chunk))[1]  # type: ignore[index]

        in_vector = chunk in vector_map
        in_bm25 = chunk in bm25_map
        if in_vector and in_bm25:
            found_by = "Both"
        elif in_vector:
            found_by = "Vector"
        else:
            found_by = "BM25"

        fused.append((chunk, rrf_score, v_score, b_score, page_num, found_by))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused