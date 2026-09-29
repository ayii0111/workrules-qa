"""問答：檢索相關條文 → 交給 LLM 依據條文回答，並標註出處。"""

import re
import sqlite3
from dataclasses import dataclass

from . import llm
from .retrieval import Hit, search

REFUSAL = "目前收錄的法規與公司規章中，找不到可以回答這個問題的條文"

SYSTEM = f"""你是公司內部的「規章與勞動法規問答助理」，回答員工、主管與人資關於請假、加班、工時、工資、離職、退休等問題。

回答規則：
1. 只能根據下方提供的「條文」回答，不要使用條文以外的知識
2. 每個重點後面標註出處，格式為【勞動基準法 第 38 條】或【工作規則 第 9 條】
3. 公司工作規則與法規都有規定時，兩者都要說明：
   - 工作規則比法規更優惠或只是補充流程 → 依工作規則
   - 工作規則比法規更不利於勞工 → 必須明確指出「此條工作規則可能違反法令」，並說明依法應如何處理
     （依勞動基準法第 1 條，雇主所訂勞動條件不得低於本法最低標準）
4. 需要計算時（例如加班費、特休天數），說明計算規則即可，並提醒使用「🧮 試算」功能取得精確數字
5. 條文無法回答問題時，以「{REFUSAL}」開頭回答，並建議詢問人資部門
6. 使用繁體中文，用一般員工看得懂的白話說明：先給結論，再列重點；不要照抄整條條文
7. 最後一行固定加上：「※ 以上說明僅供參考，實際適用仍以人資部門及主管機關解釋為準。」"""


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    provider: str | None


def build_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        prefix = "【公司工作規則】" if h.kind == "handbook" else "【法規】"
        parts.append(f"{prefix}{h.text}")
    return "\n\n".join(parts)


def ask(conn: sqlite3.Connection, question: str) -> Answer:
    hits = search(conn, question)
    if not hits:
        return Answer(REFUSAL + "。", [], None)
    text, provider = llm.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"【條文】\n{build_context(hits)}\n\n【問題】{question}"},
        ]
    )
    # 拒答時檢索到的條文本來就不相關，不列出來避免誤導
    if text.strip().startswith(REFUSAL):
        return Answer(text, [], provider)
    return Answer(text, cited_hits(text, hits), provider)


CITATION_RE = re.compile(r"【([^】]+)】")
REF_RE = re.compile(r"(.+?)\s*第\s*([\d\-]+)\s*條")


def cited_hits(text: str, hits: list[Hit]) -> list[Hit]:
    """只保留回答中實際引用的條文，順序依回答中第一次出現的位置；解析不到引用時全部保留。"""
    cited: list[tuple[str, str]] = []
    for bracket in CITATION_RE.findall(text):
        for part in re.split(r"[、，,]", bracket):
            if m := REF_RE.search(part.strip()):
                cited.append((m.group(1).strip(), m.group(2)))

    def matches(h: Hit, name: str, flno: str) -> bool:
        if h.flno != flno:
            return False
        if h.kind == "handbook":
            return "工作規則" in name or "規章" in name
        return name == h.law or name.replace("勞基法", "勞動基準法") == h.law

    ordered = []
    for name, flno in cited:
        for h in hits:
            if h not in ordered and matches(h, name, flno):
                ordered.append(h)
    return ordered or hits
