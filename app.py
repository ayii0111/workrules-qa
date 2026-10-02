"""Streamlit 介面：uv run streamlit run app.py"""

import threading
import time
from datetime import date

import pandas as pd
import streamlit as st

from workrules import calc, conversation, db, guard, llm, qa
from workrules.progress import Progress, format_stage

st.set_page_config(page_title="規章與勞動法規問答助理", page_icon="📘", layout="wide")


@st.cache_resource
def get_conn():
    # Streamlit 會在不同執行緒重跑腳本，所以關掉同執行緒檢查；本介面只讀取，不會併發寫入
    return db.connect(check_same_thread=False)


conn = get_conn()

st.title("📘 規章與勞動法規問答助理")
st.caption("員工與主管的請假、加班、特休、離職問題，依據公司工作規則與勞動法規回答，並附上條文出處。"
           "本站的公司工作規則為虛構範例。")

tab_ask, tab_calc, tab_browse, tab_status = st.tabs(["💬 問答", "🧮 試算", "📚 條文與函釋", "🗂️ 資料狀態"])

# ── 問答 ───────────────────────────────────────────────────
EXAMPLES = [
    "公司說加班只能換補休不能領錢，這樣合法嗎？",
    "我做滿一年可以放幾天特休？沒休完怎麼辦？",
    "我想離職，要提前多久跟公司說？",
    "生理假一年可以請幾天？會扣薪水嗎？",
]


TIPS = f"""
1. **一次問一個主題**；換主題請按「🆕 新對話」
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
    result: dict = {}
    history, skip = list(st.session_state.turns), skip_models()

    def work():
        try:
            result["turn"] = qa.ask(conn, question, history, skip_models=skip, progress=progress)
        except Exception as e:  # noqa: BLE001 —— 未預期的錯誤也要回到畫面，不能讓讀秒永遠轉下去
            result["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    board = st.empty()
    while worker.is_alive():
        now = time.time()
        lines = [format_stage(s, now) for s in progress.snapshot()] or ["⏳ 準備中"]
        board.markdown("  \n".join(lines) + f"  \n**總計 {progress.total():.1f} 秒**")
        time.sleep(0.2)
    board.empty()
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


with tab_ask:
    if not llm.available_providers():
        st.warning("尚未設定 LLM API key，問答功能暫時無法使用；試算與條文查詢仍可使用。")
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("ask_times", [])
    st.session_state.setdefault("cooldown", {})   # 模型 → 暫停使用到何時

    tip_col, new_col = st.columns([5, 1])
    with tip_col.expander("💡 使用小提醒", expanded=not st.session_state.turns):
        st.markdown(TIPS)
    if new_col.button("🆕 新對話", width="stretch", disabled=not st.session_state.turns):
        st.session_state.turns = []
        st.rerun()

    cols = st.columns(len(EXAMPLES))
    clicked = next((q for col, q in zip(cols, EXAMPLES) if col.button(q, width="stretch")), None)

    for t in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(t.question)
        with st.chat_message("assistant"):
            render_turn(t)

    question = st.chat_input("輸入問題，例如：家人住院需要照顧，可以請什麼假？") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(guard.mask_pii(question).text if not guard.too_long(question) else question[:80] + "…")
        with st.chat_message("assistant"):
            turn = handle(question)
            render_turn(turn)
        st.session_state.turns.append(turn)

# ── 試算 ───────────────────────────────────────────────────
with tab_calc:
    st.info("試算由程式依條文公式計算，不經過 AI，結果可重複驗證。僅供參考，實際金額以公司核算為準。")
    left, right = st.columns(2)

    with left:
        st.subheader("特別休假天數")
        st.caption("依勞動基準法第 38 條，週年制（依到職日起算）")
        start = st.date_input("到職日", value=date(2024, 3, 1), min_value=date(1980, 1, 1))
        on = st.date_input("查詢日", value=date.today())
        if on < start:
            st.error("查詢日不能早於到職日")
        else:
            months = calc.months_between(start, on)
            st.metric("目前這一段年資的特休", f"{calc.annual_leave_days(start, on)} 天",
                      help=f"年資 {months // 12} 年 {months % 12} 個月")
            sched = calc.leave_schedule(start, years=12)
            st.dataframe(
                pd.DataFrame({"取得特休的日期": [d.isoformat() for d, _ in sched],
                              "年資": ["滿 6 個月"] + [f"滿 {y} 年" for y in range(1, len(sched))],
                              "天數": [n for _, n in sched]}),
                hide_index=True, width="stretch", height=250,
            )

    with right:
        st.subheader("加班費")
        st.caption("依勞動基準法第 24 條；時薪 = 月薪 ÷ 30 ÷ 8")
        salary = st.number_input("月薪（元）", min_value=0, value=36000, step=1000)
        day_type = st.radio("加班日", ["平日", "休息日"], horizontal=True)
        max_hours = 4.0 if day_type == "平日" else 12.0
        hours = st.number_input("加班時數", min_value=0.5, max_value=max_hours, value=2.0, step=0.5)
        lines = calc.overtime_pay(salary, hours, "workday" if day_type == "平日" else "restday")
        st.metric("加班費合計", f"{sum(l.amount for l in lines):,} 元",
                  help=f"時薪 {calc.hourly_wage(salary):,.1f} 元")
        st.dataframe(
            pd.DataFrame({"時數": [l.hours for l in lines],
                          "倍率": [f"{l.rate.numerator}/{l.rate.denominator}" for l in lines],
                          "金額（元）": [l.amount for l in lines]}),
            hide_index=True, width="stretch",
        )

# ── 條文查詢 ───────────────────────────────────────────────
with tab_browse:
    srcs = conn.execute("SELECT code, name, kind, version, url FROM sources ORDER BY kind DESC, code").fetchall()
    if not srcs:
        st.info("尚無資料")
    else:
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

# ── 資料狀態 ───────────────────────────────────────────────
with tab_status:
    st.subheader("收錄範圍")
    st.dataframe(
        pd.read_sql_query(
            """SELECT s.name AS 名稱, CASE s.kind WHEN 'law' THEN '法規' WHEN 'interpretation' THEN '函釋' WHEN 'guidance' THEN '主管機關說明' ELSE '公司規章' END AS 類型,
                      s.version AS 版本, count(a.id) AS 條文數, sum(a.embedding IS NOT NULL) AS 已向量化,
                      s.fetched_at AS 更新時間
               FROM sources s LEFT JOIN articles a ON a.code = s.code GROUP BY s.code ORDER BY s.kind DESC""",
            conn,
        ),
        hide_index=True, width="stretch",
    )
    st.caption("法規每週自動檢查一次；全國法規資料庫的「修正日期」改變時才會重新抓取該部法規。")
    st.subheader("最近執行紀錄")
    runs = pd.read_sql_query(
        "SELECT started_at AS 時間, job AS 步驟, status AS 狀態, n_new AS 更新, message AS 訊息 FROM runs ORDER BY id DESC LIMIT 20",
        conn,
    )
    runs["訊息"] = runs["訊息"].fillna("").map(lambda m: m.split("\n")[0])
    st.dataframe(runs, hide_index=True, width="stretch")
