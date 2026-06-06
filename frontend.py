import streamlit as st
import requests
import time

# =====================================
# CONFIG
# =====================================

st.set_page_config(
    page_title=" RAG Assistant",
    page_icon="📚",
    layout="wide"
)

BACKEND_URL = "http://backend:8000"

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.doc-card {
    background-color: #262730;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #4CAF50;
}

.big-title {
    font-size: 2.3rem;
    font-weight: bold;
}

.small-subtitle {
    color: #999999;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown(
    """
<div class='big-title'>
📚 Vectorless RAG Assistant
</div>

<div class='small-subtitle'>
Hybrid Retrieval • OCR • Source Citations • OpenRouter + Ollama
</div>

<br>
""",
    unsafe_allow_html=True
)

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# LOAD DOCUMENTS
# =====================================

documents = []

try:

    response = requests.get(
        f"{BACKEND_URL}/documents",
        timeout=10
    )

    documents = response.json()

except:

    documents = []

# =====================================
# TOP METRICS
# =====================================

total_docs = len(documents)

total_chunks = sum(
    int(doc["chunks"])
    for doc in documents
) if documents else 0

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "📄 Indexed Documents",
        total_docs
    )

with col2:

    st.metric(
        "🧩 Stored Chunks",
        total_chunks
    )

with col3:

    st.metric(
        "💬 Messages",
        len(st.session_state.messages)
    )

st.divider()

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("⚙ Control Panel")

    # -------------------------------
    # Upload
    # -------------------------------

    st.subheader("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Supported Formats",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "xlsx",
            "json",
            "md",
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file:

        if st.button(
            "🚀 Upload",
            use_container_width=True
        ):

            with st.spinner(
                "Indexing document..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue()
                    )
                }

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files
                    )

                    data = response.json()

                    if "error" in data:

                        st.error(
                            data["error"]
                        )

                    else:

                        st.success(
                            data["message"]
                        )

                        if "chunks" in data:

                            st.info(
                                f"Chunks Indexed: {data['chunks']}"
                            )

                        time.sleep(1)

                        st.rerun()

                except Exception as e:

                    st.error(str(e))

    st.divider()

    # -------------------------------
    # Documents
    # -------------------------------

    st.subheader("📑 Indexed Documents")

    if documents:

        for doc in documents:

            col1, col2 = st.columns(
                [4, 1]
            )

            with col1:

                st.markdown(
                    f"""
<div class="doc-card">

📄 <b>{doc['filename']}</b>

<br>

Type: {doc['filetype']}

<br>

Chunks: {doc['chunks']}

</div>
""",
                    unsafe_allow_html=True
                )

            with col2:

                if st.button(
                    "❌",
                    key=f"delete_{doc['filename']}"
                ):

                    try:

                        response = requests.delete(
                            f"{BACKEND_URL}/documents/{doc['filename']}"
                        )

                        data = response.json()

                        if "error" in data:

                            st.error(
                                data["error"]
                            )

                        else:

                            st.success(
                                "Deleted"
                            )

                            time.sleep(1)

                            st.rerun()

                    except Exception as e:

                        st.error(str(e))

    else:

        st.info(
            "No documents indexed yet."
        )

    st.divider()

    # -------------------------------
    # Clear Chat
    # -------------------------------

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

# =====================================
# CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================
# CHAT INPUT
# =====================================

query = st.chat_input(
    "Ask a question about your documents..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        answer = ""

        try:

            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "query": query
                },
                stream=True
            )

            for chunk in response.iter_content(
                chunk_size=1024,
                decode_unicode=True
            ):

                if chunk:

                    answer += chunk

                    placeholder.markdown(
                        answer
                    )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(str(e))