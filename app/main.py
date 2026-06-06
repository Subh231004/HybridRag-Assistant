import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from app.parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_csv,
    extract_text_from_xlsx,
    extract_text_from_json,
    extract_text_from_md
)

from app.ocr import extract_text_from_image

from app.database import (
    conn,
    cursor
)

from app.chunker import chunk_text
from app.retriever import HybridRetriever
from app.agents import detect_task
from app.llm import stream_answer

app = FastAPI()

# =========================================
# Load Existing Chunks
# =========================================

saved_chunks = cursor.execute(
    """
    SELECT
        filename,
        page,
        chunk
    FROM documents
    """
).fetchall()

all_chunks = []

for filename, page, chunk in saved_chunks:

    all_chunks.append({

        "text": chunk,

        "source": filename,

        "page": page
    })

# =========================================
# Build Retriever
# =========================================

if len(all_chunks) > 0:

    retriever = HybridRetriever(
        all_chunks
    )

else:

    retriever = None


# =========================================
# Upload
# =========================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    global retriever
    global all_chunks

    os.makedirs(
        "data",
        exist_ok=True
    )

    file_path = os.path.join(
        "data",
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(
            await file.read()
        )

    extension = (
        file.filename
        .split(".")[-1]
        .lower()
    )

    # =====================================
    # Duplicate Detection
    # =====================================

    existing = cursor.execute(
        """
        SELECT COUNT(*)
        FROM documents
        WHERE filename=?
        """,
        (file.filename,)
    ).fetchone()[0]

    if existing > 0:

        return {

            "message":
            "File already indexed",

            "filename":
            file.filename
        }

    try:

        if extension == "pdf":

            pages = extract_text_from_pdf(
                file_path
            )

            new_chunks = []

            for page_data in pages:

                page_num = page_data["page"]

                page_text = page_data["text"]

                chunks = chunk_text(

                    page_text,

                    source=file.filename,

                    page=page_num
                )

                for chunk in chunks:

                    new_chunks.append(
                        chunk
                    )

                    all_chunks.append(
                        chunk
                    )

                    cursor.execute(
                        """
                        INSERT INTO documents
                        (
                            filename,
                            filetype,
                            page,
                            chunk
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            file.filename,
                            extension,
                            page_num,
                            chunk["text"]
                        )
                    )

            conn.commit()

            if retriever is None:

                retriever = HybridRetriever(
                    new_chunks
                )

            else:

                retriever.add_chunks(
                    new_chunks
                )

            return {

                "message":
                "PDF uploaded successfully",

                "filename":
                file.filename,

                "chunks":
                len(new_chunks)
            }

        elif extension == "docx":

            text = extract_text_from_docx(
                file_path
            )

        elif extension == "txt":

            text = extract_text_from_txt(
                file_path
            )

        elif extension == "csv":

            text = extract_text_from_csv(
                file_path
            )

        elif extension == "xlsx":

            text = extract_text_from_xlsx(
                file_path
            )

        elif extension == "json":

            text = extract_text_from_json(
                file_path
            )

        elif extension == "md":

            text = extract_text_from_md(
                file_path
            )

        elif extension in [

            "png",
            "jpg",
            "jpeg"

        ]:

            text = extract_text_from_image(
                file_path
            )

        else:

            return {

                "error":
                f"{extension} not supported"
            }

    except Exception as e:

        return {

            "error":
            str(e)
        }

    # =====================================
    # Chunk
    # =====================================

    chunks = chunk_text(

        text,

        source=file.filename,

        page=1
    )

    for chunk in chunks:

        all_chunks.append(
            chunk
        )

        cursor.execute(
            """
            INSERT INTO documents
            (
                filename,
                filetype,
                page,
                chunk
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                file.filename,
                extension,
                1,
                chunk["text"]
            )
        )

    conn.commit()

    if retriever is None:

        retriever = HybridRetriever(
            chunks
        )

    else:

        retriever.add_chunks(
            chunks
        )

    return {

        "message":
        "File uploaded successfully",

        "filename":
        file.filename,

        "chunks":
        len(chunks)
    }


# =========================================
# Documents
# =========================================

@app.get("/documents")
def list_documents():

    rows = cursor.execute(
        """
        SELECT

            filename,
            filetype,
            upload_time,
            COUNT(*)

        FROM documents

        GROUP BY filename

        ORDER BY upload_time DESC
        """
    ).fetchall()

    docs = []

    for row in rows:

        docs.append({

            "filename": row[0],

            "filetype": row[1],

            "uploaded": row[2],

            "chunks": int(row[3]),
        })

    return docs


# =========================================
# Ask
# =========================================

@app.post("/ask")
async def ask_question(data: dict):

    global retriever

    if retriever is None:

        return {
            "error": "No documents uploaded yet"
        }

    query = data["query"]

    task = detect_task(query)

    # =====================================
    # Metadata Filtering
    # =====================================

    query_lower = query.lower()

    source_filter = None

    for chunk in all_chunks:

        filename = chunk["source"]

        filename_clean = (
            filename
            .replace(".pdf", "")
            .replace(".docx", "")
            .replace(".txt", "")
            .replace(".csv", "")
            .replace(".xlsx", "")
            .replace(".json", "")
            .replace(".md", "")
            .replace("_", " ")
            .lower()
        )

        if filename_clean in query_lower:

            source_filter = filename
            break

        keywords = filename_clean.split()

        matches = sum(
            1
            for word in keywords
            if word in query_lower
        )

        if matches >= 2:

            source_filter = filename
            break

    # =====================================
    # Retrieval
    # =====================================

    relevant_chunks = retriever.retrieve(

        query,

        top_k=5,

        source_filter=source_filter
    )

    if len(relevant_chunks) == 0:

        return {
            "answer":
            "No relevant information found."
        }

    # =====================================
    # Context
    # =====================================

    context = ""

    for chunk in relevant_chunks:

        context += f"""

SOURCE: {chunk['source']}
PAGE: {chunk['page']}

{chunk['text']}

"""

    # =====================================
    # Sources
    # =====================================

    sources = []

    seen = set()

    for chunk in relevant_chunks:

        citation = (

            chunk["source"],

            chunk["page"]
        )

        if citation not in seen:

            seen.add(citation)

            sources.append(citation)

    # =====================================
    # Streaming Generator
    # =====================================

    def generate():

        full_answer = ""

        for token in stream_answer(

            query,

            context,

            task

        ):

            full_answer += token

            yield token

        # Save History

        cursor.execute(
            """
            INSERT INTO chat_history
            (
                question,
                answer
            )
            VALUES (?, ?)
            """,
            (
                query,
                full_answer
            )
        )

        conn.commit()

        # Sources

        yield "\n\n---\n\n"
        yield "### Sources\n\n"

        for source, page in sources:

            yield (
                f"- {source} "
                f"(Page {page})\n"
            )

    return StreamingResponse(

        generate(),

        media_type="text/plain"
    )


@app.delete("/documents/{filename}")
def delete_document(filename: str):

    global retriever
    global all_chunks

    cursor.execute(
        """
        DELETE FROM documents
        WHERE filename=?
        """,
        (filename,)
    )

    conn.commit()

    filepath = os.path.join(
        "data",
        filename
    )

    if os.path.exists(filepath):

        os.remove(filepath)

    all_chunks = [

        chunk

        for chunk in all_chunks

        if chunk["source"] != filename
    ]

    if len(all_chunks) > 0:

        retriever = HybridRetriever(
            all_chunks
        )

    else:

        retriever = None

    return {
        "message": f"{filename} deleted"
    }