from html import escape

import faiss, numpy as np, streamlit as st
from travelmate.embeddings import load_embedder, embed_texts
from travelmate.index_store import IndexStore
from travelmate.generator import generate_answer

st.set_page_config(page_title="TravelMate", page_icon="🧳", layout="wide")

# --- Styling & helpers ---------------------------------------------------- #
st.markdown(
    """
    <style>
    :root {
        --tm-primary: #2563eb;
        --tm-primary-dark: #1d4ed8;
        --tm-muted: #94a3b8;
        --tm-bg: #f6f8ff;
        --tm-ink: #0f172a;
        --tm-surface: #ffffff;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--tm-bg);
        padding-top: 0.5rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffffdd;
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    /* Fix text visibility for Streamlit components */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #0f172a !important;
    }
    /* Fix success/info/error messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        color: #0f172a !important;
    }
    .stSuccess p, .stInfo p, .stWarning p, .stError p {
        color: #0f172a !important;
    }
    /* Fix chat messages text color */
    [data-testid="stChatMessage"] {
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div {
        color: #0f172a !important;
    }
    /* Fix chat input text - should be white on dark background */
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
    }
    .tm-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.4rem;
        border-radius: 16px;
        background: #0f172a;
        color: #e2e8f0;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
    }
    .tm-topbar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #fff;
    }
    .tm-topbar-subtitle {
        font-size: 0.85rem;
        color: rgba(226, 232, 240, 0.85);
    }
    .tm-topbar-pill {
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tm-card {
        background: var(--tm-surface);
        border-radius: 22px;
        padding: 1.6rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.04);
        color: var(--tm-ink);
    }
    .tm-card h1,
    .tm-card h2,
    .tm-card h3,
    .tm-card p,
    .tm-card div,
    .tm-card span {
        color: var(--tm-ink);
    }
    .tm-label {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--tm-muted);
        font-weight: 600;
    }
    .tm-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(37, 99, 235, 0.09);
        color: var(--tm-primary);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .tm-status-card h2 {
        margin: 0.4rem 0 0.2rem;
    }
    .tm-status-detail {
        font-size: 0.95rem;
        color: #475569;
    }
    .tm-hint {
        background: #eef2ff;
        border: 1px solid rgba(37, 99, 235, 0.2);
        color: #1e293b;
        padding: 0.6rem 0.85rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-top: 0.75rem;
        font-weight: 500;
    }
    .tm-question-card {
        background: var(--tm-surface);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 24px 50px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.06);
        margin-top: 1.2rem;
    }
    .tm-question-card .tm-label {
        margin-bottom: 0.5rem;
    }
    [data-testid="stTextInput"] > div > div > input {
        border-radius: 12px;
        background: #f8fafc;
        border: 2px solid rgba(15, 23, 42, 0.1);
        padding: 0.65rem 0.9rem;
        color: var(--tm-ink);
        font-size: 1rem;
    }
    [data-testid="stTextInput"] > div > div > input:focus {
        border-color: var(--tm-primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    [data-testid="stForm"] {
        background: transparent;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
    }
    [data-testid="stForm"] button {
        border-radius: 999px;
        font-weight: 600;
        background: linear-gradient(90deg, #111827, #0f172a);
        color: #fff;
        border: none;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
        padding: 0.4rem 0;
        margin-top: 0.9rem;
    }
    [data-testid="stForm"] button:hover {
        background: linear-gradient(90deg, #0f172a, #111827);
    }
    /* NUCLEAR OPTION: Force white text on ALL buttons in sidebar with dark backgrounds */
    [data-testid="stSidebar"] button {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    [data-testid="stSidebar"] button span,
    [data-testid="stSidebar"] button div,
    [data-testid="stSidebar"] button p,
    [data-testid="stSidebar"] button::before,
    [data-testid="stSidebar"] button::after {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Target the specific emotion-cache classes from DevTools */
    .st-emotion-cache-g9em7g,
    .st-emotion-cache-g9em7g *,
    .ef3psqc16,
    .ef3psqc16 *,
    button.st-emotion-cache-g9em7g,
    button.st-emotion-cache-g9em7g *,
    button.ef3psqc16,
    button.ef3psqc16 * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Target buttons with the dark background color we saw in DevTools */
    button[style*="#2B2C36"],
    button[style*="2B2C36"],
    button[style*="rgb(43, 44, 54)"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* More aggressive - target any button in sidebar */
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .element-container button,
    [data-testid="stSidebar"] [class*="button"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Even more aggressive - all text inside sidebar buttons */
    [data-testid="stSidebar"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Fix subheader (Chat) text color */
    h3, h2 {
        color: #0f172a !important;
    }
    /* Fix chat input placeholder and text */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #ffffff !important;
        opacity: 0.7;
    }
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
    }
    .tm-turn {
        margin-bottom: 1rem;
    }
    .tm-turn-question,
    .tm-turn-answer {
        font-size: 1rem;
        line-height: 1.6;
        margin: 0.35rem 0 0;
    }
    .tm-citation {
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        background: #ffffff;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.06);
        height: 100%;
        color: var(--tm-ink);
    }
    .tm-code-chip {
        display: inline-flex;
        padding: 0.2rem 0.55rem;
        border-radius: 10px;
        background: rgba(37, 99, 235, 0.15);
        color: var(--tm-primary-dark);
        font-family: "JetBrains Mono", "SFMono-Regular", Menlo, monospace;
        font-size: 0.82rem;
        margin: 0.35rem 0;
    }
    .tm-muted-line {
        font-size: 0.8rem;
        color: var(--tm-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="tm-topbar">
        <div>
            <div class="tm-topbar-title">TravelMate</div>
            <div class="tm-topbar-subtitle">Grounded answers for smarter trips</div>
        </div>
        <div class="tm-topbar-pill">RAG mode</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("history", [])


def is_index_loaded() -> bool:
    return all(key in st.session_state for key in ("index", "metadatas", "texts"))


def reset_history():
    st.session_state["history"] = []


def to_html(text: str) -> str:
    """Escape text and preserve newlines for HTML blocks."""
    return "<br/>".join(escape(text).splitlines())


def render_citations(citations):
    if not citations:
        return

    st.markdown(
        '<div class="tm-label" style="margin-top:0.4rem;margin-bottom:0.5rem;">Sources</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2 if len(citations) > 1 else 1, gap="large")
    for idx, citation in enumerate(citations):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(
                f"""
                <div class="tm-citation">
                    <div class="tm-label" style="color:var(--tm-primary);">
                        Source [{citation['rank']}]
                    </div>
                    <div class="tm-code-chip">{escape(citation['source'])}</div>
                    <div class="tm-muted-line">Chunk #{citation['chunk_id']}</div>
                    <p style="font-size:0.95rem;line-height:1.5;margin-top:0.8rem;">
                        {to_html(citation['text'])}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- Sidebar -------------------------------------------------------------- #
with st.sidebar:
    st.header("Control center")
    st.caption("Connect your index, then start chatting.")
    st.markdown("Build your knowledge base with `ingest.py` before chatting.")

    index_dir = st.text_input("Index directory", "./index")
    k = st.slider("Top-k passages", min_value=1, max_value=10, value=5)

    load_clicked = st.button("Load index", use_container_width=True)
    clear_clicked = st.button("Clear conversation", use_container_width=True)

    if load_clicked:
        try:
            store = IndexStore(index_dir=index_dir)
            index, metadatas, texts = store.load()
            st.session_state["index"] = index
            st.session_state["metadatas"] = metadatas
            st.session_state["texts"] = texts
            st.success(f"Loaded index with {len(texts)} chunks.")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

    if clear_clicked:
        reset_history()
        st.info("Conversation cleared.")

    hint_text = (
        "⬆️ Load an index to unlock retrieval."
        if not is_index_loaded()
        else f"✅ Index ready · {len(st.session_state['texts'])} chunks"
    )
    st.markdown(f"<div class='tm-hint'>{hint_text}</div>", unsafe_allow_html=True)

# --- Hero & status -------------------------------------------------------- #
cols = st.columns([3, 2], gap="large")
with cols[0]:
    st.markdown(
        """
        <div class="tm-card">
            <div class="tm-pill">🧳 Travel intelligence</div>
            <h1 style="margin:0.4rem 0 0;color:var(--tm-ink);">TravelMate</h1>
            <p style="color:#475569;font-size:1rem;margin-top:0.5rem;">
                Ask natural-language questions and get grounded answers sourced
                from your curated destination knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[1]:
    status_ready = is_index_loaded()
    status = "Ready" if status_ready else "Waiting for index"
    detail = (
        f"{len(st.session_state.get('texts', []))} chunks available"
        if status_ready
        else "No index connected"
    )
    hint = "Answers will be grounded in your travel corpus." if status_ready else "Use the sidebar to load an index."
    accent = "#22c55e" if status_ready else "#f97316"
    st.markdown(
        f"""
        <div class="tm-card tm-status-card">
            <div class="tm-label">Index status</div>
            <h2 style="color:{accent};">{status}</h2>
            <p class="tm-status-detail">{detail}</p>
            <p style="color:#0f172a;font-size:0.95rem;margin-top:0.6rem;">{hint}</p>
            <div class="tm-pill" style="margin-top:1rem;">Top‑k slider adjusts recall</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Chat Interface --------------------------------------------------------- #
st.subheader("Chat")

# Display chat history
if not st.session_state["history"]:
    with st.chat_message("assistant"):
        st.write("Hello! I'm TravelMate, your travel assistant. Ask me anything about the cities in my knowledge base!")

# Display all conversation history
for turn in st.session_state["history"]:
    # User message
    with st.chat_message("user"):
        st.write(turn["question"])
    
    # Assistant message
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        # Show citations in expander
        if turn["citations"]:
            with st.expander("View sources"):
                render_citations(turn["citations"])

# Chat input (always visible at bottom)
if prompt := st.chat_input("Ask a travel question..."):
    if not is_index_loaded():
        st.warning("Please load the index in the sidebar first.")
    else:
        # Add user message to history immediately
        st.session_state["history"].append({
            "question": prompt,
            "answer": "",
            "citations": []
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embedder = load_embedder()
                
                # Simple query embedding - no conversation context
                q_emb = embed_texts(embedder, [prompt]).astype("float32")
                faiss.normalize_L2(q_emb)
                _, I = st.session_state["index"].search(q_emb, k)

                # Extract city name from question for filtering
                question_lower = prompt.lower()
                
                city_keywords = {
                    'paris': 'paris',
                    'cdg': 'paris',
                    'charles de gaulle': 'paris',
                    'orly': 'paris',
                    'ory': 'paris',
                    'rer b': 'paris',
                    'tokyo': 'tokyo',
                    'london': 'london',
                    'rome': 'rome',
                    'new york': 'newyork',
                    'nyc': 'newyork',
                    'bangkok': 'bangkok',
                    'sydney': 'sydney',
                    'dubai': 'dubai',
                    'mumbai': 'mumbai',
                    'são paulo': 'saopaulo',
                    'sao paulo': 'saopaulo',
                    'cape town': 'capetown',
                    'berlin': 'berlin',
                    'singapore': 'singapore',
                    'mexico city': 'mexicocity',
                }
                
                # Determine which city the question is about (check current question AND conversation history)
                relevant_city = None
                for keyword, city_file in city_keywords.items():
                    if keyword in conversation_text:
                        relevant_city = city_file
                        break
                
                retrieved = []
                for rank, idx in enumerate(I[0]):
                    if idx == -1:
                        continue
                    meta = st.session_state["metadatas"][idx]
                    source_file = meta["source"].lower()
                    
                    # Filter: if question mentions a specific city, prioritize chunks from that city
                    if relevant_city and relevant_city not in source_file:
                        # Skip chunks from other cities if we identified a specific city
                        continue
                    
                    retrieved.append(
                        {
                            "rank": len(retrieved) + 1,
                            "text": st.session_state["texts"][idx],
                            "source": meta["source"],
                            "chunk_id": meta["chunk_id"],
                        }
                    )
                    # For follow-up questions, get more diverse chunks
                    limit = k * 2 if history_for_context else k
                    if len(retrieved) >= limit:
                        break

                # Generate answer without conversation history
                answer = generate_answer(prompt, retrieved)
                
                # Update the last history entry with the answer
                st.session_state["history"][-1]["answer"] = answer
                st.session_state["history"][-1]["citations"] = retrieved
                
                st.write(answer)
                
                # Show citations in expander
                if retrieved:
                    with st.expander("View sources"):
                        render_citations(retrieved)
