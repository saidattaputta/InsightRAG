import json
import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_pdf
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "sample_documents" / "sample.pdf"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
TOP_K = 5


def load_questions(path: Path) -> list[dict]:
    """Load and validate the evaluation questions."""
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for item in questions:
        if "question" not in item or "expected_pages" not in item:
            raise ValueError(
                "Each question needs 'question' and 'expected_pages'."
            )

        if not isinstance(item["expected_pages"], list):
            raise ValueError("'expected_pages' must be a list.")

    return questions


def evaluate_retrieval(
    retriever: Retriever,
    questions: list[dict],
    top_k: int = TOP_K,
) -> dict:
    """Calculate page-level Recall@K and MRR for answerable questions."""
    evaluated = []
    reciprocal_ranks = []
    recalls = []

    for item in questions:
        question = item["question"]
        expected_pages = set(item["expected_pages"])

        # Unanswerable questions have no relevant page labels.
        if not item.get("answerable", True) or not expected_pages:
            print(f"\n[SKIPPED RETRIEVAL METRICS] {question}")
            print("Reason: no expected relevant pages.")
            continue

        results = retriever.search(question, top_k=top_k)
        retrieved_pages = [result["page_number"] for result in results]
        retrieved_unique_pages = set(retrieved_pages)

        relevant_retrieved = expected_pages & retrieved_unique_pages
        recall = len(relevant_retrieved) / len(expected_pages)

        first_relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result["page_number"] in expected_pages
            ),
            None,
        )

        reciprocal_rank = (
            1 / first_relevant_rank if first_relevant_rank else 0.0
        )

        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        evaluated.append(
            {
                "question": question,
                "expected_pages": sorted(expected_pages),
                "retrieved_pages": retrieved_pages,
                "recall_at_k": recall,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": reciprocal_rank,
            }
        )

        print(f"\nQuestion: {question}")
        print(f"Expected relevant pages: {sorted(expected_pages)}")
        print(f"Retrieved pages (top {top_k}): {retrieved_pages}")
        print(f"Recall@{top_k}: {recall:.3f}")
        print(f"First relevant rank: {first_relevant_rank}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")

    if not evaluated:
        raise ValueError(
            "No answerable questions with expected pages were evaluated."
        )

    return {
        "top_k": top_k,
        "evaluated_questions": len(evaluated),
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "details": evaluated,
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {QUESTIONS_PATH}"
        )

    questions = load_questions(QUESTIONS_PATH)

    print("Loading PDF...")
    pages = load_pdf(str(PDF_PATH))

    print("Creating chunks...")
    chunks = chunk_pages(pages)

    if not chunks:
        raise RuntimeError("No chunks were created from the PDF.")

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    print("\nBuilding retrieval index...")
    embedding_model = EmbeddingModel()
    retriever = Retriever(embedding_model)
    retriever.index_chunks(chunks)

    print("\nRunning retrieval evaluation...")
    report = evaluate_retrieval(retriever, questions)

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Questions evaluated: {report['evaluated_questions']}")
    print(f"Mean Recall@{TOP_K}: {report['mean_recall_at_k']:.3f}")
    print(f"MRR: {report['mrr']:.3f}")

    output_path = PROJECT_ROOT / "evaluation" / "retrieval_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
