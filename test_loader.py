from src.document_loader import load_pdf

pdf_path = 'data/sample_documents/sample.pdf'

pages = load_pdf(pdf_path)

print(f"Extracted {len(pages)} pages from {pdf_path}.")

for page in pages[:3]:
    print(f'\nsource: {page["source"]}, page_number: {page["page_number"]}, is_empty: {page["is_empty"]}, text_preview: {page["text"][:500]}')
    