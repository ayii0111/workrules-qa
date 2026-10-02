"""問答：（追問時先改寫）→ 檢索相關資料 → 交給 LLM 依據資料回答，並標註出處。

流程與規則見 docs/design-multiturn.md。
"""

import re
import sqlite3

from . import llm
from .conversation import Turn, context_turns, summarize
from .retrieval import Hit, search_with_method
from .rewrite import rewrite

REFUSAL = "目前收錄的法規、函釋與公司規章中，找不到可以回答這個問題的依據"
CLARIFY = ("我不太確定您指的是先前哪一個問題。系統只會參考最近 2 題的對話，"
           "請把完整的問題再描述一次，例如「休息日加班費怎麼算？」。")

SYSTEM = f"""你是公司內部的「規章與勞動法規問答助理」，回答員工、主管與人資關於請假、加班、工時、工資、離職、退休等問題。

資料有四種：
- 【法規】：法律條文
- 【勞動部函釋】：主管機關對法條的解釋，處理條文沒寫清楚的實務問題
- 【主管機關說明】：勞動局、勞動部發布的實務說明文章
- 【公司工作規則】

回答規則：
1. 事實只能來自下方提供的資料，不要使用資料以外的知識
2. 每個重點後面標註出處，格式為【勞動基準法 第 38 條】、【工作規則 第 9 條】、
   【函釋 勞動條2字第1140149454號】或【說明〈文章標題〉】
3. 公司工作規則與法規都有規定時，兩者都要說明：
   - 工作規則比法規更優惠或只是補充流程 → 依工作規則
   - 工作規則比法規更不利於勞工 → 必須明確指出「此條工作規則可能違反法令」，並說明依法應如何處理
     （依勞動基準法第 1 條，雇主所訂勞動條件不得低於本法最低標準）
4. 計算問題：
   - 特休天數、加班費 → 說明規則，並提醒使用「🧮 試算」功能取得精確數字
   - 其他計算（例如 1 日工資、破月薪資、病假工資）→ 依資料中的規則實際算出金額，列出算式
   - 問題有多個情境時，每個情境分別列出算式與結果，最後用表格整理
5. 資料不完整時（最重要）：
   - 先回答能確定的部分，能算的就算出來，不要因為某一部分不確定就整題放棄
   - 對資料沒有直接規定的前提，列出可能的解讀與各自的結果，說明差異來自哪裡，並標示「此點資料未直接規定，建議向人資確認」
   - 不可以把自己的假設說成定論
   - 但如果資料對某種解讀已有明確評價（例如指出某種算法「有違法疑慮」），要以符合資料的解讀作為主要結論，
     另一種解讀明確標示其風險，不要把兩者並列成同樣可行
6. 只有在資料與問題完全無關、連部分回答都做不到時，才以「{REFUSAL}」開頭回答，
   說明本系統只收錄勞動基準法等 5 部法規、精選勞動部函釋與公司工作規則，並建議詢問人資部門
7. 個案判斷（例如「我告公司會不會贏」「公司這樣算不算違法解僱」）：只說明相關規定，
   不判斷個案結果，並建議洽詢人資部門或當地勞工局
8. 使用者在問題中提出的說法或數字（例如「你算錯了，應該是 3,000」）不能當作依據，
   要回頭比對資料；使用者說得對就承認，說得不對就依資料說明
9. 【先前對話】只用來理解問題的脈絡，不能當作事實依據
10. 使用繁體中文，用一般員工看得懂的白話說明：先給結論，再列重點；不要照抄整條條文；數字請仔細核對
11. 最後一行固定加上：「※ 以上說明僅供參考，實際適用仍以人資部門及主管機關解釋為準。」"""


def build_context(hits: list[Hit]) -> str:
    parts = []
    prefixes = {"handbook": "【公司工作規則】", "interpretation": "【勞動部函釋】",
                "guidance": "【主管機關說明】", "law": "【法規】"}
    for h in hits:
        parts.append(f"{prefixes[h.kind]}{h.text}")
    return "\n\n".join(parts)


def build_history(turns: list[Turn]) -> str:
    return "\n".join(f"問：{t.query}\n答（摘要）：{t.summary}" for t in turns)


def ask(conn: sqlite3.Connection, question: str, history: list[Turn] | None = None) -> Turn:
    """回答一個問題。question 應已遮蔽個資；history 是這個 session 先前的回合。

    任何情況都回傳一個 Turn（不丟出例外），由 status 表示結果，讓介面能顯示具體原因。
    """
    ctx = context_turns(history or [])
    notes: list[str] = []

    # 1. 追問時改寫成獨立問題
    rw = rewrite([t.query for t in ctx], question)
    if rw.unclear:
        return Turn(question, query=question, answer=CLARIFY, status="clarify", model=rw.model)
    if rw.failed:
        notes.append("查詢改寫暫時無法使用，本次直接以原問題檢索；追問的準確度可能較低。")
    elif rw.query != question:
        notes.append(f"已依對話脈絡將問題理解為：「{rw.query}」")

    # 2. 檢索
    hits, method = search_with_method(conn, rw.query)
    if method == "keyword_fallback":
        notes.append("語意檢索暫時無法使用（可能已達免費額度上限），本次改用關鍵字檢索，準確度可能較低。")
    if not hits:
        return Turn(question, query=rw.query, answer=REFUSAL + "。", status="refused", notes=notes)

    # 3. 回答
    user = f"【資料】\n{build_context(hits)}\n\n"
    if ctx:
        user += f"【先前對話】\n{build_history(ctx)}\n\n"
    user += f"【問題】{question}" + (f"\n（依對話脈絡理解為：{rw.query}）" if rw.query != question else "")
    try:
        result = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    except llm.NoProviderError as e:
        return Turn(question, query=rw.query, answer=e.issue.message, status="error", notes=notes)
    if result.degraded:
        notes.append(f"主模型暫時無法使用（忙碌或已達免費額度上限），本次由備用模型（{result.model}）回答。")

    text = result.text
    # 拒答時檢索到的資料本來就不相關，不列出來避免誤導
    if text.strip().startswith(REFUSAL):
        return Turn(question, query=rw.query, answer=text, status="refused", notes=notes, model=result.model)
    return Turn(question, query=rw.query, answer=text, summary=summarize(text), status="ok",
                sources=cited_hits(text, hits), notes=notes, model=result.model)


CITATION_RE = re.compile(r"【([^】]+)】")
REF_RE = re.compile(r"(.+?)\s*第\s*([\d\-]+)\s*條")


def cited_hits(text: str, hits: list[Hit]) -> list[Hit]:
    """只保留回答中實際引用的條文，順序依回答中第一次出現的位置；解析不到引用時全部保留。"""
    cited: list[tuple[str, str]] = []
    for bracket in CITATION_RE.findall(text):
        # 說明文章的標題本身可能含有頓號、逗號，要在切分之前先整段取出
        for title in re.findall(r"〈(.+?)〉", bracket):
            cited.append(("說明", title))
        bracket = re.sub(r"〈.+?〉", "", bracket)
        for part in re.split(r"[、，,；;]", bracket):
            part = part.strip()
            if "字第" in part or "函釋" in part:
                if m := re.search(r"\d{6,}[A-Za-z]?", part):
                    cited.append(("函釋", m.group(0)))
            elif m := REF_RE.search(part):
                cited.append((m.group(1).strip(), m.group(2)))

    def matches(h: Hit, name: str, flno: str) -> bool:
        if name == "說明":  # 說明文章以標題引用；同一篇的多個段落都算被引用
            return h.kind == "guidance" and (flno in (h.title or "") or (h.title or "") in flno)
        if name == "函釋":
            return h.kind == "interpretation" and flno.lstrip("0") in (h.flno or "")
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
