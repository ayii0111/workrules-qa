"""SQLite 資料庫：schema 與連線。

選 SQLite 的理由：單一檔案、免架伺服器，clone 下來就能跑；
法規加上公司規章只有幾百條，效能綽綽有餘。
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
-- 每部法規（或公司規章）一列，記錄版本日期，用來判斷是否需要重新抓取
CREATE TABLE IF NOT EXISTS sources (
    code        TEXT PRIMARY KEY,             -- 法規代碼（pcode），公司規章為 HANDBOOK
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,                -- law / handbook
    version     TEXT NOT NULL,                -- 法規的修正日期；規章為檔案內容雜湊
    url         TEXT,
    fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 一條條文一列：法規本來就是以「條」為單位組織，照條切塊比固定字數切塊更完整
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY,
    code        TEXT NOT NULL REFERENCES sources(code) ON DELETE CASCADE,
    flno        TEXT NOT NULL,                -- 條號，如 38、10-1
    title       TEXT,                         -- 條文標題（公司規章才有）
    chapter     TEXT,
    content     TEXT NOT NULL,
    url         TEXT,
    embedding   BLOB,                         -- float32 向量；沒有 embedding 服務時為 NULL
    UNIQUE (code, flno)
);

-- 全文檢索：存 jieba 斷詞後以空白分隔的字串，讓 FTS5 能處理中文
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(tokens, article_id UNINDEXED);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    status      TEXT NOT NULL,                -- ok / error
    n_new       INTEGER NOT NULL DEFAULT 0,
    message     TEXT
);
"""


def connect(path: Path | None = None, *, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: Path | None = None):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_run(conn: sqlite3.Connection, job: str, status: str, n_new: int = 0, message: str | None = None):
    conn.execute(
        "INSERT INTO runs (job, status, n_new, message) VALUES (?, ?, ?, ?)",
        (job, status, n_new, message),
    )
