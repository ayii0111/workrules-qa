"""建立檢索索引：條文 → 全文檢索 → 補上向量。"""

import sqlite3

import numpy as np

from . import llm
from .text import tokenize


def article_text(row: sqlite3.Row) -> str:
    """檢索與 LLM 看到的條文文字：帶上法規名稱、條號、章名。

    單看條文內容常常不知道出自哪部法規（例如「前項規定…」），
    帶上這些資訊後，檢索和 LLM 都比較能判斷，也方便引用。
    """
    head = f"{row['name']} 第 {row['flno']} 條"
    if row["title"]:
        head += f"（{row['title']}）"
    if row["chapter"]:
        head += f"｜{row['chapter']}"
    return f"{head}\n{row['content']}"


ARTICLE_SQL = """SELECT a.id, a.code, a.flno, a.title, a.chapter, a.content, a.url, s.name, s.kind
                 FROM articles a JOIN sources s ON s.code = a.code"""


def index_missing(conn: sqlite3.Connection) -> int:
    """替還沒進全文索引的條文建立索引。"""
    rows = conn.execute(
        ARTICLE_SQL + " WHERE a.id NOT IN (SELECT article_id FROM articles_fts)"
    ).fetchall()
    conn.executemany(
        "INSERT INTO articles_fts (tokens, article_id) VALUES (?, ?)",
        [(" ".join(tokenize(article_text(r))), r["id"]) for r in rows],
    )
    return len(rows)


def embed_missing(conn: sqlite3.Connection, batch_size: int = 90) -> int:
    """替還沒有向量的條文補上向量；沒有 embedding 服務時什麼都不做。

    每批完成就先寫入，中途因額度中斷時，已完成的部分不會白做。
    """
    rows = conn.execute(ARTICLE_SQL + " WHERE a.embedding IS NULL").fetchall()
    done = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        vectors = llm.embed([article_text(r) for r in batch])
        if vectors is None:
            break
        conn.executemany(
            "UPDATE articles SET embedding = ? WHERE id = ?",
            [(v.astype(np.float32).tobytes(), r["id"]) for v, r in zip(vectors, batch)],
        )
        conn.commit()
        done += len(batch)
    return done
