"""條文檢索。

預設策略（auto）是依評估結果決定的，見 docs/evaluation.md：
1. 問題直接寫出條號（「勞基法第 84-1 條」）→ 那一條直接排第一，不靠檢索去猜
2. 其餘用語意向量檢索：員工會問「做滿一年可以放幾天假」，條文寫的卻是「一年以上二年未滿者，七日」，
   兩者幾乎沒有共同的詞，關鍵字檢索找不到
3. 向量服務不可用（額度用完、服務中斷）→ 退回關鍵字檢索，至少還能用

原本設計為關鍵字＋向量的混合檢索（業界常見做法），但在這批「員工口語提問」上實測，
關鍵字結果大多是雜訊，混進來反而把正確條文擠下去；關鍵字權重越低結果越好，所以改為上述策略。
混合模式保留下來，供評估比較使用。

向量檢索用 numpy 暴力計算餘弦相似度，沒有另外架向量資料庫：
條文只有幾百條，全部算一遍不到幾毫秒。
"""

import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import llm
from .index import ARTICLE_SQL, article_text
from .text import tokenize

RRF_K = 60  # RRF 常用預設值，用來降低單一排名第一名的權重
MODES = ("auto", "hybrid", "keyword", "vector")


@dataclass
class Hit:
    article_id: int
    code: str
    law: str
    kind: str  # law / handbook
    flno: str
    title: str | None
    text: str
    url: str | None
    score: float

    @property
    def label(self) -> str:
        if self.kind == "interpretation":
            return f"勞動部函釋 {self.title}"
        if self.kind == "guidance":
            # 同一篇文章切成多段，標籤帶上段落名稱才分得出來
            section = self.text.split("\n", 1)[0].rsplit("｜", 1)[-1]
            return f"主管機關說明〈{self.title}〉{section}"
        name = "工作規則" if self.kind == "handbook" else self.law
        return f"{name} 第 {self.flno} 條" + (f"（{self.title}）" if self.title else "")


def keyword_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[int]:
    tokens = tokenize(query)
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
    rows = conn.execute(
        "SELECT article_id FROM articles_fts WHERE articles_fts MATCH ? ORDER BY bm25(articles_fts) LIMIT ?",
        (match, limit),
    ).fetchall()
    return [r["article_id"] for r in rows]


_QUERY_VECTORS: dict[str, np.ndarray] = {}
_QUERY_CACHE_SIZE = 256


def query_vector(query: str) -> np.ndarray | None:
    """問題轉向量，結果快取：同一句話（例如評估時對同一題比較多種檢索方式）只呼叫一次 API。
    失敗的結果不快取，下次會重試。"""
    if query in _QUERY_VECTORS:
        return _QUERY_VECTORS[query]
    q = llm.embed([query], max_waits=0)  # 使用者在等，額度不足時不等待，直接退回關鍵字
    if q is not None:
        if len(_QUERY_VECTORS) >= _QUERY_CACHE_SIZE:
            _QUERY_VECTORS.pop(next(iter(_QUERY_VECTORS)))
        _QUERY_VECTORS[query] = q
    return q


def vector_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[int]:
    rows = conn.execute("SELECT id, embedding FROM articles WHERE embedding IS NOT NULL").fetchall()
    if not rows:
        return []
    q = query_vector(query)
    if q is None:
        return []
    matrix = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    if matrix.shape[1] != q.shape[1]:
        return []  # 換過 embedding 模型、維度不符時，寧可不用也不要算出錯誤結果
    scores = matrix @ q[0]
    return [rows[i]["id"] for i in np.argsort(-scores)[:limit]]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion：每個結果的分數 = Σ 1/(k + 名次)。

    只看名次、不看原始分數，所以 BM25 與餘弦相似度這兩種尺度不同的分數可以直接合併。
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, aid in enumerate(ranking, start=1):
            scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    return scores


ARTICLE_REF_RE = re.compile(r"第\s*([\d\-]+)\s*條")


def explicit_refs(conn: sqlite3.Connection, query: str) -> list[int]:
    """問題裡直接寫「勞基法第 38 條」時，直接把那一條排第一，不靠檢索去猜。"""
    aliases = {"勞動基準法": ["勞基法"], "性別平等工作法": ["性平法"], "勞工退休金條例": ["勞退條例"]}
    ids = []
    # 函釋文號（例如「勞動條2字第1140149454號」）：比對文號中的數字
    for digits in re.findall(r"\d{6,}[A-Za-z]?", query):
        row = conn.execute(
            "SELECT a.id FROM articles a JOIN sources s ON s.code = a.code "
            "WHERE s.kind = 'interpretation' AND a.flno LIKE ?", (f"%{digits}%",)).fetchone()
        if row:
            ids.append(row["id"])
    for flno in ARTICLE_REF_RE.findall(query):
        sql = "SELECT a.id, s.name, s.kind FROM articles a JOIN sources s ON s.code = a.code WHERE a.flno = ?"
        for r in conn.execute(sql, (flno,)):
            if r["kind"] == "handbook":
                names = ["工作規則", "規章", "內規"]
            else:
                names = [r["name"], *aliases.get(r["name"], [])]
            # 「勞動基準法第 38 條」不能誤配到「勞動基準法施行細則第 38 條」
            if any(n in query for n in names) and not (r["name"] == "勞動基準法" and "施行細則" in query):
                ids.append(r["id"])
    return ids


def rank_with_method(conn: sqlite3.Connection, query: str, mode: str = "auto") -> tuple[list[int], str]:
    """回傳 (排序後的 id, 實際使用的檢索方式)。auto 模式下向量不可用時，方式會是 keyword_fallback。"""
    if mode == "keyword":
        return keyword_search(conn, query), "keyword"
    if mode == "vector":
        return vector_search(conn, query), "vector"
    if mode == "hybrid":
        scores = rrf([keyword_search(conn, query), vector_search(conn, query)])
        return sorted(scores, key=scores.get, reverse=True), "hybrid"
    ranked, method = vector_search(conn, query), "vector"
    if not ranked:
        ranked, method = keyword_search(conn, query), "keyword_fallback"
    pinned = explicit_refs(conn, query)
    return pinned + [a for a in ranked if a not in pinned], method


def rank(conn: sqlite3.Connection, query: str, mode: str = "auto") -> list[int]:
    return rank_with_method(conn, query, mode)[0]


def search(conn: sqlite3.Connection, query: str, top_k: int = 8, mode: str = "auto") -> list[Hit]:
    return search_with_method(conn, query, top_k, mode)[0]


def search_with_method(conn: sqlite3.Connection, query: str, top_k: int = 8,
                       mode: str = "auto") -> tuple[list[Hit], str]:
    ranked, method = rank_with_method(conn, query, mode)
    best = ranked[:top_k]
    if not best:
        return [], method
    rows = {r["id"]: r for r in conn.execute(
        ARTICLE_SQL + f" WHERE a.id IN ({','.join('?' * len(best))})", best)}
    hits = [
        Hit(aid, rows[aid]["code"], rows[aid]["name"], rows[aid]["kind"], rows[aid]["flno"],
            rows[aid]["title"], article_text(rows[aid]), rows[aid]["url"], 1.0 / (i + 1))
        for i, aid in enumerate(best)
    ]
    return hits, method
