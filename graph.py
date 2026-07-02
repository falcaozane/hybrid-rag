"""
LangGraph orchestration for the Hybrid RAG pipeline.

Graph flow (query path):
    START → retrieve_vector → retrieve_bm25 → fuse_results → generate_answer → END

parse_and_index() is a standalone helper called from app.py on PDF upload —
it is NOT a graph node because FAISS/BM25 index construction is a one-time
side-effect, not part of the per-query retrieval flow.

Architecture note on session_state:
    FAISS and BM25 indexes are large, non-JSON-serialisable objects.
    They live in st.session_state, not in RAGState.
    Each graph node is a closure that captures session_state at build time,
    so nodes can read the indexes without polluting the state schema.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# Load .env from the project root (one directory up from this file, or same dir).
# override=False so a real env var already set in the shell takes precedence.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

import anthropic
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    pdf_text: list[str]                  # raw chunks (unused in query flow)
    query: str
    vector_results: list[tuple]          # (chunk, score, page_num)
    bm25_results: list[tuple]            # (chunk, score, page_num)
    fused_chunks: list[tuple]            # (chunk, rrf_score, v_score, b_score, page_num, found_by)
    answer: str
    prompt_sent: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Standalone indexing helper (called from app.py, not a graph node)
# ---------------------------------------------------------------------------

def parse_and_index(pdf_files: list, session_state: dict) -> None:
    """
    Extract, chunk, and index all uploaded PDFs into a single FAISS + BM25 index.
    Results are written into session_state so they persist across Streamlit reruns.

    pdf_files: list of Streamlit UploadedFile objects.
    IMPORTANT: .read() is called once per file — the stream is consumed.
    """
    from indexer.pdf_indexer import (
        build_bm25_index,
        build_faiss_index,
        chunk_text,
        extract_pdf,
    )

    all_chunks: list[str] = []
    all_chunk_pages: list[int] = []
    file_metadata: list[dict] = []

    for uploaded_file in pdf_files:
        pdf_bytes = uploaded_file.read()  # consume stream exactly once
        pages = extract_pdf(pdf_bytes)

        if not pages:
            # PDF has no extractable text (e.g. scanned image-only PDF)
            file_metadata.append(
                {
                    "filename": uploaded_file.name,
                    "page_count": 0,
                    "chunk_count": 0,
                }
            )
            continue

        chunks, chunk_pages = chunk_text(pages)
        all_chunks.extend(chunks)
        all_chunk_pages.extend(chunk_pages)
        file_metadata.append(
            {
                "filename": uploaded_file.name,
                "page_count": len(pages),
                "chunk_count": len(chunks),
            }
        )

    if not all_chunks:
        # All PDFs were image-only — nothing to index
        session_state["chunks"] = []
        session_state["chunk_pages"] = []
        session_state["faiss_index"] = None
        session_state["bm25_index"] = None
        session_state["file_metadata"] = file_metadata
        session_state["indexed"] = False
        return

    session_state["faiss_index"] = build_faiss_index(all_chunks)
    session_state["bm25_index"] = build_bm25_index(all_chunks)
    session_state["chunks"] = all_chunks
    session_state["chunk_pages"] = all_chunk_pages
    session_state["file_metadata"] = file_metadata
    session_state["indexed"] = True


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(session_state: dict, top_k: int = 5, retrieval_mode: str = "Both"):
    """
    Build and compile a LangGraph StateGraph for one query invocation.

    Called fresh on every Send click so the current top_k / retrieval_mode
    values are always captured. Graph construction has negligible cost.

    retrieval_mode:
        "Both"   — run FAISS + BM25, fuse with RRF  (default)
        "Vector" — run FAISS only; BM25 node returns []
        "BM25"   — run BM25 only;  FAISS node returns []
    """

    # ------------------------------------------------------------------
    # Node: retrieve_vector
    # ------------------------------------------------------------------
    def retrieve_vector_fn(state: RAGState) -> dict:
        if retrieval_mode == "BM25":
            # BM25-only mode — skip vector retrieval entirely
            return {"vector_results": []}

        from retriever.vector_retriever import retrieve

        results = retrieve(
            query=state["query"],
            faiss_index=session_state["faiss_index"],
            chunks=session_state["chunks"],
            chunk_pages=session_state["chunk_pages"],
            k=top_k,
        )
        return {"vector_results": results}

    # ------------------------------------------------------------------
    # Node: retrieve_bm25
    # ------------------------------------------------------------------
    def retrieve_bm25_fn(state: RAGState) -> dict:
        if retrieval_mode == "Vector":
            # Vector-only mode — skip BM25 retrieval entirely
            return {"bm25_results": []}

        from retriever.bm25_retriever import retrieve

        results = retrieve(
            query=state["query"],
            bm25_index=session_state["bm25_index"],
            chunks=session_state["chunks"],
            chunk_pages=session_state["chunk_pages"],
            k=top_k,
        )
        return {"bm25_results": results}

    # ------------------------------------------------------------------
    # Node: fuse_results
    # ------------------------------------------------------------------
    def fuse_results_fn(state: RAGState) -> dict:
        from retriever.fusion import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion(
            vector_results=state["vector_results"],
            bm25_results=state["bm25_results"],
            rrf_k=60,
        )
        return {"fused_chunks": fused}

    # ------------------------------------------------------------------
    # Node: generate_answer
    # ------------------------------------------------------------------
    def generate_answer_fn(state: RAGState) -> dict:
        top_chunks = state["fused_chunks"][:top_k]

        # Build context block — each chunk prefixed with its page number
        context_parts = [
            f"[Page {chunk[4]}]\n{chunk[0]}" for chunk in top_chunks
        ]
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a precise and thorough research assistant specializing in "
            "document question-answering. Your task is to answer the user's question "
            "using ONLY the information contained in the provided context passages.\n\n"
            "Guidelines:\n"
            "- Base your answer strictly on the provided context. Do NOT use outside knowledge.\n"
            "- Cite the relevant page number(s) inline using the format [Page N] wherever you "
            "draw information from a specific passage.\n"
            "- If multiple passages support the same point, cite all relevant pages.\n"
            "- If the context contains partial but useful information, share what is available "
            "and clearly note what is missing or unclear.\n"
            "- If the context does not contain enough information to answer the question, "
            "state this explicitly and explain what kind of information would be needed.\n"
            "- Keep your answer focused and well-structured. Use bullet points or numbered "
            "lists when presenting multiple distinct pieces of information.\n"
            "- Do not speculate, infer beyond what the text supports, or fabricate details."
        )

        user_message = (
            f"Here are the relevant context passages retrieved from the document(s):\n\n"
            f"{context}\n\n"
            f"---\n\n"
            f"Question: {state['query']}\n\n"
            f"Please answer the question based solely on the context above, citing page "
            f"numbers where applicable."
        )

        prompt = f"{system_prompt}\n\n{user_message}"  # kept for prompt_sent logging

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)

        t0 = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        answer = response.content[0].text
        usage = response.usage

        return {
            "answer": answer,
            "prompt_sent": prompt,
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Assemble graph
    # ------------------------------------------------------------------
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_vector", retrieve_vector_fn)
    graph.add_node("retrieve_bm25", retrieve_bm25_fn)
    graph.add_node("fuse_results", fuse_results_fn)
    graph.add_node("generate_answer", generate_answer_fn)

    graph.add_edge(START, "retrieve_vector")
    graph.add_edge("retrieve_vector", "retrieve_bm25")
    graph.add_edge("retrieve_bm25", "fuse_results")
    graph.add_edge("fuse_results", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Graph visualisation helper
# ---------------------------------------------------------------------------

def save_graph_image(output_path: str | Path = "graph.png") -> Path:
    """
    Render the RAG pipeline graph and save it as a PNG image.

    Uses LangGraph's built-in Mermaid renderer (no external dependencies
    beyond what langgraph already ships with).

    Args:
        output_path: Destination file path (default: ``graph.png`` next to
                     this source file).

    Returns:
        The resolved :class:`~pathlib.Path` of the saved file.

    Example::

        from graph import save_graph_image
        save_graph_image("docs/rag_graph.png")
    """
    output_path = Path(output_path)

    # Build a minimal graph with placeholder session_state — we only need the
    # topology, so no real FAISS / BM25 indexes are required.
    compiled = build_graph(session_state={}, top_k=5, retrieval_mode="Both")

    # draw_mermaid_png() returns raw PNG bytes; write them to disk.
    png_bytes: bytes = compiled.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_bytes)

    print(f"Graph image saved → {output_path.resolve()}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry-point: python graph.py [output_path]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    dest = sys.argv[1] if len(sys.argv) > 1 else "graph.png"
    save_graph_image(dest)