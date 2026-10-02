"""Streamlit 介面：uv run streamlit run app.py"""

import threading
import time

import streamlit as st

from workrules import conversation, db, guard, llm, qa
from workrules.progress import Progress, format_stage

st.set_page_config(page_title="規章與勞動法規問答助理", page_icon="📘", layout="wide")


@st.cache_resource
def get_conn():
    # Streamlit 會在不同執行緒重跑腳本，所以關掉同執行緒檢查；本介面只讀取，不會併發寫入
    return db.connect(check_same_thread=False)


conn = get_conn()

# ── 問答 ───────────────────────────────────────────────────
EXAMPLES = [
    "公司說加班只能換補休不能領錢，這樣合法嗎？",
    "我做滿一年可以放幾天特休？沒休完怎麼辦？",
    "月薪三萬，月中上兩天班就離職，薪水怎麼算？",
    "颱風天沒去上班，公司可以扣全勤獎金嗎？",
]


TIPS = f"""
1. **一次問一個主題**；換主題請按右上角「🗑️ 清除對話」
2. 追問只會參考**最近 {conversation.WINDOW} 題**；重新整理頁面會清空對話
3. **請勿輸入**姓名、身分證字號、薪資明細等個人資料（系統會自動遮蔽身分證、手機與 Email）
4. 收錄範圍：勞動基準法等 5 部法規、精選勞動部函釋與主管機關說明、公司工作規則（虛構範例）。個案爭議請洽人資或勞工局
5. 本 demo 使用 Google Gemini **免費方案**，每分鐘與每日都有使用上限；達到上限時畫面會說明需等待多久
"""


def render_turn(turn: conversation.Turn):
    """依狀態顯示：錯誤與被擋下用醒目的提示框，讓使用者一眼知道發生什麼事。"""
    if turn.status in ("error", "blocked"):
        st.warning(turn.answer)
    elif turn.status == "clarify":
        st.info(turn.answer)
    else:
        st.markdown(turn.answer)
    if turn.sources:
        with st.expander(f"📎 引用資料（{len(turn.sources)}）"):
            for h in turn.sources:
                st.markdown(f"**{h.label}**" + (f"　[原文]({h.url})" if h.url else ""))
                st.caption(h.text.split("\n", 1)[-1])
    for note in turn.notes:
        st.caption(f"ℹ️ {note}")
    if turn.timeline:
        steps = "　→　".join(
            f"{'❌ ' if s.state == 'failed' else ''}{s.label} {s.seconds():.1f}s" + (f"（{s.note}）" if s.note else "")
            for s in turn.timeline)
        st.caption(f"⏱ 共 {turn.total_seconds:.1f} 秒：{steps}")


# 模型失敗後，這個 session 先跳過它多久（秒）。只存在 st.session_state，不與其他使用者共用
COOLDOWN = {"timeout": 300, "overloaded": 300, "unavailable": 300, "rate_day": 3600}


def remember_failures(failures):
    now = time.time()
    for model, issue in failures:
        seconds = (issue.retry_seconds or 60) if issue.kind == "rate_minute" else COOLDOWN.get(issue.kind)
        if seconds:
            st.session_state.cooldown[model] = now + seconds


def skip_models() -> frozenset[str]:
    now = time.time()
    return frozenset(m for m, until in st.session_state.cooldown.items() if until > now)


def run_with_timer(question: str) -> conversation.Turn:
    """問答放在背景執行緒，畫面每 0.2 秒重畫一次各階段的讀秒。

    呼叫 API 時 Streamlit 的畫面會凍結，所以不能在主執行緒等；背景執行緒只更新 Progress 物件，
    不呼叫任何 Streamlit 函式（Streamlit 的函式只能在主執行緒使用）。
    """
    progress = Progress()
    result: dict = {"partial": ""}
    history, skip = list(st.session_state.turns), skip_models()

    def on_text(text: str):
        result["partial"] = text  # 背景執行緒只寫入資料，由主執行緒負責畫面

    def work():
        try:
            result["turn"] = qa.ask(conn, question, history, skip_models=skip, progress=progress, on_text=on_text)
        except Exception as e:  # noqa: BLE001 —— 未預期的錯誤也要回到畫面，不能讓讀秒永遠轉下去
            result["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    board, answer_box = st.empty(), st.empty()
    while worker.is_alive():
        now = time.time()
        lines = [format_stage(s, now) for s in progress.snapshot()] or ["⏳ 準備中"]
        board.caption("  \n".join(lines) + f"  \n總計 {progress.total():.1f} 秒")
        if result["partial"]:
            answer_box.markdown(result["partial"] + " ▌")
        else:
            answer_box.empty()
        time.sleep(0.2)
    board.empty()
    answer_box.empty()
    if "error" in result:
        return conversation.Turn(question, answer=f"⚠️ 系統發生未預期的錯誤：{result['error']}", status="error")
    return result["turn"]


def handle(question: str) -> conversation.Turn:
    """輸入防護 → 問答。防護擋下的回合也會顯示，但不會進入之後的對話脈絡（見 conversation.py）。"""
    if guard.too_long(question):
        return conversation.Turn(question[:80] + "…", answer=(
            f"✂️ 問題超過 {guard.MAX_CHARS} 字，請精簡成一個具體的問題再送出；"
            "不需要貼上整份契約或對話紀錄，描述關鍵情況即可。"), status="blocked")
    masked = guard.mask_pii(question)
    wait = guard.wait_seconds(st.session_state.ask_times)
    if wait:
        return conversation.Turn(masked.text, answer=(
            f"⏳ 提問速度較快，為避免超過 Google Gemini 免費方案的每分鐘上限，請約 {wait} 秒後再送出。"),
            status="blocked")
    st.session_state.ask_times = guard.record(st.session_state.ask_times)
    turn = run_with_timer(masked.text)
    remember_failures(turn.failures)
    if masked.found:
        turn.notes.insert(0, f"已自動遮蔽您輸入的{'、'.join(masked.found)}，請勿在提問中提供個人資料。")
    return turn


def page_chat():
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("ask_times", [])
    st.session_state.setdefault("cooldown", {})   # 模型 → 暫停使用到何時

    # 輸入框放在頁面最外層，Streamlit 才會把它固定在畫面底部；呼叫位置不影響顯示位置
    question = st.chat_input("輸入問題，例如：家人住院需要照顧，可以請什麼假？")
    question = question or st.session_state.pop("pending", None)

    if not llm.available_providers():
        st.warning("尚未設定 LLM API key，問答功能暫時無法使用；條文與函釋仍可查詢。")

    if not st.session_state.turns and not question:
        # 空白對話：歡迎畫面與範例問題；開始對話後就不再顯示
        st.title("📘 規章與勞動法規問答助理")
        st.markdown("請假、加班、特休、薪資、離職的問題，依據**勞動法規、勞動部函釋與公司工作規則**回答，並附上出處。")
        st.caption("本站的公司工作規則為虛構範例。回答僅供參考，實際適用以人資部門及主管機關解釋為準。")
        st.write("")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, width="stretch", key=f"example_{i}"):
                st.session_state.pending = q
                st.rerun()
        st.write("")
        with st.expander("💡 使用小提醒"):
            st.markdown(TIPS)
        return

    for t in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(t.question)
        with st.chat_message("assistant"):
            render_turn(t)

    if question:
        with st.chat_message("user"):
            st.markdown(guard.mask_pii(question).text if not guard.too_long(question) else question[:80] + "…")
        with st.chat_message("assistant"):
            turn = handle(question)
            render_turn(turn)
        st.session_state.turns.append(turn)


# ── 條文與函釋 ─────────────────────────────────────────────
def page_browse():
    st.title("📚 條文與函釋")
    srcs = conn.execute("SELECT code, name, kind, version, url FROM sources ORDER BY kind DESC, code").fetchall()
    if not srcs:
        st.info("尚無資料")
        return
    names = {s["name"]: s for s in srcs}
    choice = st.selectbox("資料來源", list(names))
    keyword = st.text_input("篩選條文內容（例如：特別休假）")
    src = names[choice]
    is_interp = src["kind"] in ("interpretation", "guidance")
    origin = {"law": f"[全國法規資料庫原文]({src['url']})", "interpretation": "來源：勞動部勞動法令查詢系統",
              "guidance": "來源：各地勞動主管機關發布之說明（依其轉載規定註明出處）",
              "handbook": "虛構範例"}[src["kind"]]
    st.caption(("精選清單版本" if is_interp else "版本") + f"：{src['version']}　｜　{origin}")
    rows = conn.execute(
        "SELECT flno, title, chapter, content, url FROM articles WHERE code = ? ORDER BY id", (src["code"],)
    ).fetchall()
    rows = [r for r in rows if not keyword or keyword in r["content"] or keyword in (r["title"] or "")
            or keyword in (r["chapter"] or "")]
    st.caption(f"共 {len(rows)} " + ("則" if is_interp else "條"))
    for r in rows:
        if src["kind"] == "interpretation":
            date, topic = (r["chapter"] or "｜").split("｜", 1)
            title = f"{topic}　｜　{r['title']}（{date}）"
        elif src["kind"] == "guidance":
            meta, section = (r["chapter"] or "｜").split("｜", 1)
            title = f"〈{r['title']}〉{section}　｜　{meta}"
        else:
            title = f"第 {r['flno']} 條" + (f"　{r['title']}" if r["title"] else "") + (f"　｜　{r['chapter']}" if r["chapter"] else "")
        with st.expander(title, expanded=bool(keyword)):
            st.text(r["content"])
            if is_interp and r["url"]:
                st.markdown(f"[原文]({r['url']})")


# ── 頂端切換 ───────────────────────────────────────────────
# 不用側邊欄，也不用分頁（tabs）：聊天輸入框放在分頁裡時無法固定在畫面底部。
# 改用頂端切換鈕，問答頁的內容（含輸入框）都在頁面最外層，輸入框才能固定在底部
VIEWS = ["💬 問答", "📚 條文與函釋"]
st.session_state.setdefault("turns", [])
nav, action = st.columns([4, 1], vertical_alignment="center")
view = nav.segmented_control("頁面", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed")
view = view or st.session_state.get("last_view", VIEWS[0])  # 再點一次已選的項目會取消選取，視為不變
st.session_state.last_view = view

if view == VIEWS[0]:
    # 不依對話是否為空來停用按鈕：按鈕在本輪問答「之前」就畫好，若依當時狀態停用，
    # 第一題回答完後按鈕仍是灰的，要等下一次操作才會恢復。空的時候按下去也無害
    if action.button("🗑️ 清除對話", width="stretch", help="清空目前的對話紀錄，重新開始提問"):
        st.session_state.turns = []
        st.rerun()
    page_chat()
else:
    page_browse()
