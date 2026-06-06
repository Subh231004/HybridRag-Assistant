import sqlite3

conn = sqlite3.connect(
    "rag.db",
    check_same_thread=False
)

cursor = conn.cursor()

# =====================================
# DOCUMENTS
# =====================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    filename TEXT,

    filetype TEXT,

    page INTEGER,

    chunk TEXT,

    chunk_hash TEXT UNIQUE
)
""")

# =====================================
# CHAT HISTORY
# =====================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    question TEXT,

    answer TEXT
)
""")

conn.commit()