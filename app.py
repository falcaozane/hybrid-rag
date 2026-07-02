"""
Hybrid RAG — Streamlit UI
Sidebar: Documents panel | Main area: Chat interface
"""

from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

from graph import build_graph, parse_and_index
from monitoring.chunk_monitor import log_query

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "chunks": [],
    "chunk_pages": [],
    "chroma_collection": None,
    "bm25_index": None,
    "file_metadata": [],
    "indexed": False,
    "messages": [],
    "monitor_history": [],
    "pending_query": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Helper: render Logs tab content for one assistant message
# ---------------------------------------------------------------------------
def _render_logs(msg: dict) -> None:
    """Render the Logs tab for a single assistant message."""

    # ---- Retrieval mode badge ----
    mode = msg.get("retrieval_mode", "Both")
    _mode_color = {"Both": "🟢", "Vector": "🔵", "BM25": "🟠"}
    st.markdown(f"{_mode_color.get(mode, '⚪')} **Retrieval Mode:** `{mode}`")

    # ---- Top K Chunks table ----
    fused = msg.get("fused_chunks", [])
    if fused:
        st.markdown("##### Top K Chunks")
        rows = []
        for rank, (chunk, rrf, vscore, bscore, page, found_by) in enumerate(fused, 1):
            rows.append(
                {
                    "Rank": rank,
                    "Chunk Preview": chunk[:120],
                    "Page": page,
                    "Vector Score": round(vscore, 4),
                    "BM25 Score": round(bscore, 4),
                    "RRF Score": round(rrf, 6),
                    "Found By": found_by,
                }
            )
        df = pd.DataFrame(rows)

        def _color_found_by(val: str) -> str:
            return {
                "Both":   "background-color: #d4edda; color: #155724",
                "Vector": "background-color: #cce5ff; color: #004085",
                "BM25":   "background-color: #fff3cd; color: #856404",
            }.get(val, "")

        styled = df.style.map(_color_found_by, subset=["Found By"])
        st.dataframe(styled, width="stretch", hide_index=True)
    else:
        st.info("No chunks retrieved.")

    # ---- Prompt sent to LLM ----
    st.markdown("##### Prompt Sent to LLM")
    prompt_text = html.escape(msg.get("prompt_sent", ""))
    st.markdown(
        f"""<div style="
                height: 260px;
                overflow-y: auto;
                background: #0e1117;
                color: #fafafa;
                padding: 12px 16px;
                border-radius: 6px;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 13px;
                line-height: 1.55;
                white-space: pre-wrap;
                word-break: break-word;
                border: 1px solid #31333f;
            ">{prompt_text}</div>""",
        unsafe_allow_html=True,
    )

    # ---- Token usage ----
    st.markdown("##### Token Usage")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input Tokens",  msg.get("prompt_tokens", 0))
    c2.metric("Output Tokens", msg.get("completion_tokens", 0))
    c3.metric("Total Tokens",  msg.get("total_tokens", 0))

    # ---- Latency ----
    st.markdown("##### Latency")
    st.metric("LLM Call Time", f"{msg.get('latency_ms', 0.0):.1f} ms")


# ===========================================================================
# SIDEBAR — Document Panel
# ===========================================================================
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show Index button only when files are uploaded but not yet indexed
    if uploaded_files:
        uploaded_names = {f.name for f in uploaded_files}
        indexed_names  = {m["filename"] for m in st.session_state.file_metadata}
        needs_indexing = uploaded_names != indexed_names

        if needs_indexing:
            if st.button("⚡ Index Documents", type="primary", use_container_width=True):
                with st.spinner("Indexing PDFs… this may take a moment."):
                    parse_and_index(uploaded_files, st.session_state)

                if not st.session_state.indexed:
                    st.warning(
                        "⚠️ No extractable text found in the uploaded PDF(s). "
                        "Scanned image-only PDFs are not supported."
                    )
                st.rerun()

    # File metadata table (transposed: files as columns, metrics as rows)
    if st.session_state.file_metadata:
        meta_df = pd.DataFrame(st.session_state.file_metadata)
        meta_df.columns = ["Filename", "Pages", "Chunks"]
        meta_df["Status"] = "Indexed ✅" if st.session_state.indexed else "⚠️ No text"
        transposed = meta_df.set_index("Filename").astype(str).T
        st.dataframe(transposed, width="stretch")

    st.divider()

    # Retrieval type selector
    retrieval_mode = st.selectbox(
        "Retrieval Type",
        options=["Both", "Vector", "BM25"],
        index=0,
        help=(
            "**Both** — ChromaDB vector search + BM25 keyword search, "
            "merged with Reciprocal Rank Fusion.\n\n"
            "**Vector** — dense semantic search only (ChromaDB).\n\n"
            "**BM25** — sparse keyword search only."
        ),
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        options=["NVIDIA", "Cerebras", "Groq"],
        index=0,
    )

    # Top-K slider
    top_k = st.slider("Top K Chunks", min_value=3, max_value=10, value=5, step=1)



# ===========================================================================
# MAIN AREA — Chat Interface
# ===========================================================================
st.header("💬 Chat")

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
messages = st.session_state.messages
i = 0
while i < len(messages):
    user_msg = messages[i]
    with st.chat_message("user"):
        st.markdown(user_msg["content"])

    if i + 1 < len(messages):
        asst_msg = messages[i + 1]
        with st.chat_message("assistant"):
            tab_answer, tab_logs = st.tabs(["Answer", "Logs"])
            with tab_answer:
                st.markdown(asst_msg.get("answer", ""))
            with tab_logs:
                _render_logs(asst_msg)

    i += 2

# ---------------------------------------------------------------------------
# Chat input + Send
# ---------------------------------------------------------------------------
if st.session_state.pop("clear_input", False):
    prefill = ""
else:
    prefill = st.session_state.pop("pending_query", None) or ""

input_col, btn_col = st.columns([5, 1])
with input_col:
    user_input = st.text_input(
        "chat_input_label",
        value=prefill,
        key="chat_input",
        label_visibility="collapsed",
        placeholder="Ask a question about your documents…",
    )
with btn_col:
    send_clicked = st.button("Send ➤", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if send_clicked and user_input.strip():
    if not st.session_state.indexed:
        st.warning("⚠️ Please upload and index at least one PDF first.")
    else:
        api_key_env = f"{llm_provider.upper()}_API_KEY"
        if not os.environ.get(api_key_env):
            st.error(f"❌ {api_key_env} environment variable is not set.")
        else:
            with st.spinner("Thinking…"):
                compiled_graph = build_graph(
                    st.session_state, top_k=top_k, retrieval_mode=retrieval_mode, llm_provider=llm_provider
                )

            initial_state: dict = {
                "pdf_text":          [],
                "query":             user_input.strip(),
                "vector_results":    [],
                "bm25_results":      [],
                "fused_chunks":      [],
                "answer":            "",
                "prompt_sent":       "",
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "total_tokens":      0,
                "latency_ms":        0.0,
            }

            try:
                result = compiled_graph.invoke(initial_state)
            except Exception as exc:
                st.error(f"❌ Error during generation: {exc}")
                st.stop()

        # Persist to chat history
        st.session_state.messages.append(
            {"role": "user", "content": user_input.strip()}
        )
        st.session_state.messages.append(
            {
                "role":              "assistant",
                "answer":            result["answer"],
                "fused_chunks":      result["fused_chunks"],
                "prompt_sent":       result["prompt_sent"],
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":      result["total_tokens"],
                "latency_ms":        result["latency_ms"],
                "retrieval_mode":    retrieval_mode,
            }
        )

        log_query(
            session_state=st.session_state,
            query=user_input.strip(),
            answer=result["answer"],
            fused_chunks=result["fused_chunks"],
            prompt_sent=result["prompt_sent"],
            token_counts={
                "input":  result["prompt_tokens"],
                "output": result["completion_tokens"],
                "total":  result["total_tokens"],
            },
            latency_ms=result["latency_ms"],
        )

        st.session_state["clear_input"] = True
        st.rerun()
