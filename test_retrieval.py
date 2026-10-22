
from src.document_loader import load_pdf
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever


PDF_PATH = "data/sample_documents/sample.pdf"

# 1. Extract PDF pages
pages = load_pdf(PDF_PATH)

# 2. Split pages into chunks
chunks = chunk_pages(pages)

if not chunks:
    raise RuntimeError("No chunks were produced from the PDF.")

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

# 3. Generate embeddings and build the FAISS index
embedding_model = EmbeddingModel()
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

print(f"Vectors indexed: {retriever.index.ntotal}")

# 4. Ask a question about your actual document
question = input("\nAsk a question about your PDF: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 5. Retrieve the top 5 passages
results = retriever.search(question, top_k=5)

for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page_number']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(result["text"][:600])
