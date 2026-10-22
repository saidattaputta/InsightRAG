from transformers import AutoTokenizer 

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

def chunk_pages(
        pages: list[dict],
        chunk_size: int = 600,
        overlap: int = 100,
) -> list[dict]:
    
    # Split extracted PDF pages into overlapping token chunks

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size.")

    chunks = []
    step = chunk_size - overlap

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue  # Skip empty pages
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        for start in range(0,len(token_ids), step):
            chunk_ids = token_ids[start: start + chunk_size]

            if not chunk_ids:
                break  # No more tokens to process

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
            ).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "token_count": len(chunk_ids),
                })

            if start + chunk_size >= len(token_ids):
                break  # Reached the end of the token list
            
    return chunks