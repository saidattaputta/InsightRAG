
from src.document_loader import load_pdf
from src.chunking import chunk_pages, tokenizer
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator

PDF_PATH = "data/sample_documents/sample.pdf"

# 1. Extract PDF pages
pages = load_pdf(PDF_PATH)

# 2. Split pages into chunks
chunks = chunk_pages(pages, chunk_size=450, overlap=75)

if not chunks:
    raise RuntimeError("No chunks were produced from the PDF.")

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

# Verify chunk lengths, including special tokens
longest_chunk = max(
    len(tokenizer.encode(chunk["text"], add_special_tokens=True))
    for chunk in chunks
)
print(f"Longest chunk including special tokens: {longest_chunk}")

if longest_chunk > 512:
    raise ValueError(
        "A chunk exceeds the embedding model's 512-token limit. "
        "Please adjust chunking.py."
    )

# 3. Generate embeddings and build the FAISS index
embedding_model = EmbeddingModel()
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

print(f"Vectors indexed: {retriever.index.ntotal}")

# 4. Ask a question about your document
question = input("\nAsk a question about your PDF: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 5. Retrieve the top 5 passages
results = retriever.search(question, top_k=5)

# 6. Generate a grounded answer
generator = AnswerGenerator()
answer = generator.generate_answer(question, results)

print(f"\nGenerated Answer:\n{answer}")

# 7. Display retrieved evidence
for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page_number']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(result["text"][:600])
