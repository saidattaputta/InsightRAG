from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_pdf
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator

PDF_PATH = PROJECT_ROOT / "data" / "sample_documents" / "sample.pdf"

pages = load_pdf(str(PDF_PATH))
chunks = chunk_pages(pages)

embedding_model = EmbeddingModel()
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

generator = AnswerGenerator()

test_questions = [
    "What is a Physics-Informed Neural Network (PINN)?",
    "What is the price of the battery discussed in this document?",
]

for question in test_questions:
    results = retriever.search(question, top_k=5)
    answer = generator.generate_answer(question, results)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("\nGENERATED ANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")
    for result in results:
        print(
            f"Page {result['page_number']} "
            f"(similarity={result['similarity_score']:.3f})"
        )