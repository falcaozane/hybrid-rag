"""
Query monitor — stores the last 5 query/response entries in session state.

Deliberately does NOT import streamlit; accepts session_state as a plain
dict-like object so it can be unit-tested without a Streamlit context.
"""

from __future__ import annotations

from datetime import datetime


def log_query(
    session_state: dict,
    query: str,
    answer: str,
    fused_chunks: list[tuple],
    prompt_sent: str,
    token_counts: dict,
    latency_ms: float,
) -> None:
    """
    Append a query entry to session_state['monitor_history'].

    token_counts should be: {"input": int, "output": int, "total": int}
    Keeps only the last 5 entries.
    """
    if "monitor_history" not in session_state:
        session_state["monitor_history"] = []

    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "query": query,
        # Truncate answer preview to 200 chars for compact display
        "answer": answer[:200] + "…" if len(answer) > 200 else answer,
        "chunks": fused_chunks,
        "prompt_sent": prompt_sent,
        "tokens": token_counts,
        "latency_ms": latency_ms,
    }

    session_state["monitor_history"].append(entry)

    # Enforce 5-entry cap (trim oldest)
    if len(session_state["monitor_history"]) > 5:
        session_state["monitor_history"] = session_state["monitor_history"][-5:]


def get_history(session_state: dict) -> list[dict]:
    """Return the current monitor history (up to 5 entries)."""
    return session_state.get("monitor_history", [])