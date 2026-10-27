import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_pdf
from src.chunking import chunk_pages
from src.embeddings import EmbeddingModel
from src.retriever import Retriever
from src.generator import AnswerGenerator


PDF_PATH = PROJECT_ROOT / "data" / "sample_documents" / "sample.pdf"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "answer_quality_report.json"

# Basic indicators of a refusal / abstention.
ABSTENTION_PHRASES = [
    "not available",
    "not specified",
    "does not specify",
    "doesn't specify",
    "cannot be determined",
    "insufficient information",
    "not mentioned",
    "not provided",
]


def has_citation(answer):
    """Check for a citation containing a page number."""
    pattern = r"\[[^\]]+\bp\.\s*\d+[^\]]*\]"
    return bool(re.search(pattern, answer, re.IGNORECASE))


def appears_to_abstain(answer):
    """Heuristic check for common abstention wording."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def main():
    print("Loading document...")
    pages = load_pdf(str(PDF_PATH))

    print("Creating chunks...")
    chunks = chunk_pages(pages)

    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    print("Building retriever...")
    retriever = Retriever(embedding_model)
    retriever.index_chunks(chunks)

    print("Loading generator...")
    generator = AnswerGenerator()

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    results = []

    for item in questions:
        question = item["question"]
        answerable = item["answerable"]

        print(f"\nEvaluating: {question}")

        retrieved_chunks = retriever.search(question, top_k=5)
        answer = generator.generate_answer(question, retrieved_chunks)

        citation_present = has_citation(answer)
        abstained = appears_to_abstain(answer)

        if answerable:
            passed = citation_present and not abstained
            checks = {
                "citation_present": citation_present,
                "abstained": abstained,
            }
        else:
            passed = abstained
            checks = {
                "abstained": abstained,
            }

        result = {
            "question": question,
            "answerable": answerable,
            "answer": answer,
            "checks": checks,
            "passed": passed,
        }

        results.append(result)

        print(f"Answer: {answer}")
        print(f"Checks: {checks}")
        print(f"Passed heuristic checks: {passed}")

    passed_count = sum(result["passed"] for result in results)
    total_count = len(results)

    report = {
        "total_questions": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": passed_count / total_count if total_count else 0,
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== Evaluation Summary ==========")
    print(f"Total questions: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Pass rate: {report['pass_rate']:.2%}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()