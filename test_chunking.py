from src.document_loader import load_pdf
from src.chunking import chunk_pages

pdf_path = 'data/sample_documents/sample.pdf'

pages = load_pdf(pdf_path)
chunks = chunk_pages(pages)

print(f"Extracted {len(pages)} pages from {pdf_path}.")
print(f"Created {len(chunks)} chunks.")

for chunk in chunks[:3]:
    print(f'\nchunk_id: {chunk["chunk_id"]}, source: {chunk["source"]}, page_number: {chunk["page_number"]}, token_count: {chunk["token_count"]}, text_preview: {chunk["text"][:300]}')