"""檢索品質評估：用一組「員工口語提問 → 應該找到的條文」比較三種檢索方式。

RAG 的回答品質主要取決於檢索：條文沒被找到，LLM 再強也答不對。
所以改動檢索方式前後都要跑一次，用數字判斷有沒有變好，而不是憑感覺。
"""

import json
import sqlite3

from . import config
from .retrieval import MODES, rank

EVAL_PATH = config.ROOT / "data" / "eval_set.json"


def load_cases() -> list[dict]:
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def article_keys(conn: sqlite3.Connection, ids: list[int]) -> list[str]:
    if not ids:
        return []
    rows = {r["id"]: f"{r['code']}:{r['flno']}" for r in conn.execute(
        f"SELECT id, code, flno FROM articles WHERE id IN ({','.join('?' * len(ids))})", ids)}
    return [rows[i] for i in ids if i in rows]


def evaluate(conn: sqlite3.Connection, mode: str, k: int = 8) -> dict:
    """hit@3 / hit@k：前 3 / 前 k 筆有沒有包含任一預期條文。MRR：第一個命中條文名次的倒數平均。"""
    cases = load_cases()
    hit3 = hitk = rr = 0.0
    misses = []
    for c in cases:
        keys = article_keys(conn, rank(conn, c["q"], mode)[:k])
        pos = next((i for i, key in enumerate(keys, start=1) if key in c["expect"]), None)
        hit3 += bool(pos and pos <= 3)
        hitk += bool(pos)
        rr += 1 / pos if pos else 0
        if not pos:
            misses.append(c["q"])
    n = len(cases)
    return {"mode": mode, "hit@3": hit3 / n, f"hit@{k}": hitk / n, "MRR": rr / n, "misses": misses}


def report(conn: sqlite3.Connection) -> str:
    names = {"keyword": "只用關鍵字", "vector": "只用向量", "hybrid": "混合（關鍵字＋向量，RRF）",
             "auto": "本系統（條號直達＋向量，失敗退回關鍵字）"}
    lines = [f"共 {len(load_cases())} 題", "", "| 檢索方式 | hit@3 | hit@8 | MRR |", "|---|---|---|---|"]
    results = [evaluate(conn, m) for m in ("keyword", "vector", "hybrid", "auto") if m in MODES]
    for r in results:
        lines.append(f"| {names[r['mode']]} | {r['hit@3']:.0%} | {r['hit@8']:.0%} | {r['MRR']:.2f} |")
    for r in results:
        if r["misses"]:
            lines.append(f"\n{names[r['mode']]} 沒找到：" + "、".join(r["misses"]))
    return "\n".join(lines)
