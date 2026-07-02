"""
Dense (ChromaDB) retriever.
"""

from __future__ import annotations

def retrieve(
    query: str,
    chroma_collection,
    k: int = 5,
) -> list[tuple[str, float, int]]:
    """
    Search the ChromaDB collection, return top-k results.

    Returns list of (chunk_text, distance_score, page_num), sorted by score.
    """
    results = chroma_collection.query(
        query_texts=[query],
        n_results=k
    )

    formatted_results: list[tuple[str, float, int]] = []
    
    # Chroma returns lists of lists for queries
    if not results["documents"] or not results["documents"][0]:
        return formatted_results
        
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        dist = results["distances"][0][i] if results["distances"] else 0.0
        page = results["metadatas"][0][i]["page"] if results["metadatas"] else 0
        formatted_results.append((doc, float(dist), int(page)))

    return formatted_results