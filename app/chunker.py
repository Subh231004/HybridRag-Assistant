import hashlib


def chunk_text(
    text,
    source,
    page,
    chunk_size=800,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunk_hash = hashlib.md5(
            chunk.encode()
        ).hexdigest()

        chunks.append({

            "text": chunk,

            "source": source,

            "page": page,

            "hash": chunk_hash
        })

        start += chunk_size - overlap

    return chunks