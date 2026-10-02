"""對話紀錄：每一輪的資料結構，以及「哪些內容可以進入下一輪的脈絡」的規則。

設計見 docs/design-multiturn.md。重點：
- 只有狀態為 ok 的回合才算有效紀錄；錯誤、被擋下、需澄清的回合只顯示，不傳給 LLM
- 只取最近 WINDOW 輪，每輪成本固定，不隨對話長度成長
- 傳給下一輪的是「查詢句」與「回答摘要」，不是回答全文
"""

import re
from dataclasses import dataclass, field

WINDOW = 2
SUMMARY_CHARS = 200

# 狀態：ok 正常回答／refused 拒答／clarify 需澄清／blocked 被擋下／error API 錯誤
CONTEXT_STATUSES = {"ok"}


@dataclass
class Turn:
    question: str                 # 遮蔽個資後的問題（畫面顯示用）
    query: str = ""               # 改寫後的獨立問題（檢索與下一輪改寫用）
    answer: str = ""              # 回答全文（畫面顯示用）
    summary: str = ""             # 回答摘要（下一輪脈絡用）
    status: str = "ok"
    sources: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # 系統狀態說明：降級、改用關鍵字等
    model: str | None = None


def context_turns(turns: list[Turn], window: int = WINDOW) -> list[Turn]:
    """可以進入下一輪脈絡的回合：最近 window 個狀態為 ok 的回合。"""
    return [t for t in turns if t.status in CONTEXT_STATUSES][-window:]


def summarize(answer: str, limit: int = SUMMARY_CHARS) -> str:
    """取回答開頭的結論部分當摘要：去掉 Markdown 符號與免責聲明，截到 limit 字。

    不另外呼叫 LLM 產生摘要：回答規則要求「先給結論」，開頭本身就是結論，省一次呼叫。
    """
    text = re.sub(r"※.*", "", answer)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)   # 清單符號
    text = re.sub(r"[#*>|`]+", "", text)                    # 標題、粗體、引用、表格符號直接刪除
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")
