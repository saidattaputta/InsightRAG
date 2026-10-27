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

st.title("📚 InsightRAG")
st.subheader("AI Research Copilot")
st.write(
    "Upload a PDF, ask questions, and get answers grounded "
    "in your document with page citations."
)


# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# Sidebar: document upload
with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    process_button = st.button(
        "Process Document",
        disabled=uploaded_file is None,
    )

    if process_button and uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded PDF temporarily
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load and chunk document
                pages = load_pdf(temp_path)
                chunks = chunk_pages(pages)

                # Initialize embedding model and retriever
                embedding_model = EmbeddingModel()
                retriever = Retriever(embedding_model)
                retriever.index_chunks(chunks)

                # Initialize answer generator
                generator = AnswerGenerator()

                # Save objects in Streamlit session
                st.session_state.retriever = retriever
                st.session_state.generator = generator
                st.session_state.document_name = uploaded_file.name
                st.session_state.messages = []

                st.success(
                    f"Processed {uploaded_file.name} "
                    f"into {len(chunks)} chunks."
                )

                os.remove(temp_path)

            except Exception as exc:
                st.error(f"Document processing failed: {exc}")


# Main chat interface
if st.session_state.retriever is None:
    st.info("Upload and process a PDF from the sidebar to begin.")
else:
    st.success(
        f"Document loaded: {st.session_state.document_name}"
    )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("View retrieved passages"):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']} — "
                                f"Page {source['page_number']}**"
                            )
                            st.write(source["text"])

    # Question input
    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:
        # Display and store user message
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    retrieved_chunks = (
                        st.session_state.retriever.search(
                            question,
                            top_k=5,
                        )
                    )

                    answer = (
                        st.session_state.generator.generate_answer(
                            question,
                            retrieved_chunks,
                        )
                    )

                    st.markdown(answer)

                    with st.expander("View retrieved passages"):
                        for chunk in retrieved_chunks:
                            st.markdown(
                                f"**{chunk['source']} — "
                                f"Page {chunk['page_number']}**"
                            )
                            st.write(chunk["text"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks,
                    })

                except Exception as exc:
                    st.error(f"An error occurred: {exc}")