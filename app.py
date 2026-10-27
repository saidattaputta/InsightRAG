
import os
import tempfile

import streamlit as st

from src.document_loader import load_pdf
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator


st.set_page_config(
    page_title="InsightRAG | AI Research Copilot",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_pages" not in st.session_state:
    st.session_state.document_pages = 0

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📚 InsightRAG")
    st.caption("AI Research Copilot")

    st.divider()

    st.subheader("Document Status")

    if st.session_state.retriever is not None:
        st.success("Document ready")

        st.write(f"**File:** {st.session_state.document_name}")
        st.write(f"**Pages:** {st.session_state.document_pages}")
        st.write(f"**Chunks:** {st.session_state.document_chunks}")
    else:
        st.info("No document loaded yet.")

    st.divider()

    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Retrieved passages",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document passages used to answer each question.",
    )

    st.divider()

    if st.button("🗑️ Clear document and chat", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.generator = None
        st.session_state.messages = []
        st.session_state.document_name = None
        st.session_state.document_pages = 0
        st.session_state.document_chunks = 0

        st.rerun()

    st.divider()

    with st.expander("About InsightRAG"):
        st.write(
            "InsightRAG is an AI Research Copilot that answers "
            "questions using uploaded research documents."
        )

        st.markdown(
            """
            **Features**
            - PDF document processing
            - Semantic search
            - Grounded answers
            - Page citations
            - Evidence inspection
            """
        )

        st.caption(
            "Built with Python, Streamlit, Sentence Transformers, "
            "FAISS, and Gemini."
        )


# --------------------------------------------------
# Main Page Header
# --------------------------------------------------

st.title("InsightRAG")
st.markdown(
    "### Your AI research copilot"
)

st.write(
    "Upload a research paper or PDF, ask questions, and explore "
    "answers grounded in the document with page citations."
)


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

st.subheader("📄 Upload your document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a research paper, report, or other PDF document.",
)

if uploaded_file is not None:
    st.caption(f"Selected file: {uploaded_file.name}")

    process_button = st.button(
        "⚙️ Process document",
        type="primary",
        use_container_width=False,
    )

    if process_button:
        temp_path = None

        try:
            with st.spinner(
                "Processing PDF, creating embeddings, and building search index..."
            ):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load PDF
                pages = load_pdf(temp_path)

                # Chunk document
                chunks = chunk_pages(pages)

                if not chunks:
                    st.error(
                        "No readable text was found in this PDF. "
                        "Please try another document."
                    )
                    st.stop()

                # Create embedding model
                embedding_model = EmbeddingModel()

                # Build retriever
                retriever = Retriever(embedding_model)
                retriever.index_chunks(chunks)

                # Create generator
                generator = AnswerGenerator()

                # Save processed document to session
                st.session_state.retriever = retriever
                st.session_state.generator = generator
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_pages = len(pages)
                st.session_state.document_chunks = len(chunks)
                st.session_state.messages = []

            st.success(
                f"Successfully processed **{uploaded_file.name}** "
                f"into {len(chunks)} searchable chunks."
            )

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


st.divider()


# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

st.subheader("💬 Ask your document")

if st.session_state.retriever is None:
    st.info(
        "Upload and process a PDF above to start asking questions."
    )

else:
    st.caption(
        f"Currently chatting with: {st.session_state.document_name}"
    )

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("🔎 View retrieved passages"):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']} — "
                                f"Page {source['page_number']}**"
                            )

                            st.write(source["text"])

                            st.divider()

    # User question
    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:
        # Store and display user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate grounded answer
        with st.chat_message("assistant"):
            with st.spinner(
                "Searching document and generating answer..."
            ):
                try:
                    retrieved_chunks = (
                        st.session_state.retriever.search(
                            question,
                            top_k=top_k,
                        )
                    )

                    answer = (
                        st.session_state.generator.generate_answer(
                            question,
                            retrieved_chunks,
                        )
                    )

                    st.markdown(answer)

                    with st.expander("🔎 View retrieved passages"):
                        for chunk in retrieved_chunks:
                            st.markdown(
                                f"**{chunk['source']} — "
                                f"Page {chunk['page_number']}**"
                            )

                            st.write(chunk["text"])

                            st.divider()

                    # Save assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks,
                        }
                    )

                except Exception as exc:
                    st.error(f"An error occurred: {exc}")
