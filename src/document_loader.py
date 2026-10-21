from pathlib import Path
import re 

import fitz #PyMuPDF

def clean_text(text: str) -> str:
    "Normalize and clean text by removing extra whitespace and unwanted characters."

    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract readable text page by page.

    Each returned record contains the source filename,
    1-based page number, and cleaned page text."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages = []

    with fitz.open(pdf_path) as document:
        
        if document.needs_pass:
            raise ValueError("PDF file is password protected. Please provide a valid PDF file.")

        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))

            # preserve page numbers, including pages with no extractable text.
            pages.append({
                "source": pdf_path.name,
                "page_number": page_number,
                "text": text,
                "is_empty": not bool(text),
            })

    return pages