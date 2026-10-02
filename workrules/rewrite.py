"""查詢改寫：把追問改寫成不需要上下文也看得懂的獨立問題，再拿去檢索。

只在有對話紀錄時才呼叫；第一個問題直接檢索，不增加成本。
改寫只看過去的「查詢句」，不看回答全文：查詢句短而乾淨，成本低，也比較不會被冗長的回答帶偏。
"""

import json
import re
from dataclasses import dataclass, field

from . import llm
from .progress import Progress

PROMPT = """你負責把員工在對話中的「最新問題」改寫成一句不需要上下文也看得懂的獨立問題，用來搜尋勞動法規。

先前的問題（由舊到新）：
{history}

最新問題：{question}

判斷規則：
1. 最新問題本身已經完整、看得懂（包含換了新話題的情況）→ 原樣輸出，不要把先前問題的主題加進去
2. 最新問題是追問（例如「那休息日呢？」「會扣錢嗎？」）→ 補上先前問題的主題，改寫成完整問題
3. 最新問題指涉的對象在先前問題中找不到（例如「回到第一題」但看不到第一題）→ unclear 設為 true
4. 改寫時只補主題，不要加入先前問題中沒有出現的條件或數字

只輸出 JSON：{{"query": "改寫後的問題", "followup": true 或 false, "unclear": true 或 false}}"""


@dataclass
class Rewrite:
    query: str
    followup: bool = False
    unclear: bool = False
    model: str | None = None
    failed: bool = False  # 改寫失敗時退回原問題
    failures: list = field(default_factory=list)  # 失敗的模型與原因，交給呼叫端記錄


def parse(text: str, question: str) -> Rewrite:
    """解析 LLM 輸出；格式不對時退回原問題，而不是讓整個問答失敗。"""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError
        return Rewrite(query, bool(data.get("followup")), bool(data.get("unclear")))
    except (ValueError, json.JSONDecodeError):
        return Rewrite(question, failed=True)


def rewrite(history_queries: list[str], question: str, *, skip: frozenset[str] = frozenset(),
            progress: Progress | None = None) -> Rewrite:
    if not history_queries:
        return Rewrite(question)
    history = "\n".join(f"- {q}" for q in history_queries)
    try:
        result = llm.chat([{"role": "user", "content": PROMPT.format(history=history, question=question)}],
                          temperature=0, purpose="rewrite", skip=skip, progress=progress, stage_label="理解問題")
    except llm.NoProviderError as e:
        r = Rewrite(question, failed=True)
        r.failures = e.failures
        return r
    r = parse(result.text, question)
    r.model, r.failures = result.model, result.failures
    return r
