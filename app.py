"""Streamlit 介面：uv run streamlit run app.py"""

from datetime import date

import pandas as pd
import streamlit as st

from workrules import calc, db, llm, qa

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


def render_answer(ans: qa.Answer):
    st.markdown(ans.text)
    if ans.sources:
        with st.expander(f"📎 引用條文（{len(ans.sources)}）"):
            for h in ans.sources:
                head = f"**{h.label}**" + (f"　[原文]({h.url})" if h.url else "")
                st.markdown(head)
                st.caption(h.text.split("\n", 1)[-1])
    if ans.provider:
        st.caption(f"模型：{ans.provider}")


with tab_ask:
    if not llm.available_providers():
        st.warning("尚未設定 LLM API key，問答功能暫時無法使用；試算與條文查詢仍可使用。")
    if "history" not in st.session_state:
        st.session_state.history = []

    cols = st.columns(len(EXAMPLES))
    clicked = next((q for col, q in zip(cols, EXAMPLES) if col.button(q, width="stretch")), None)

    for role, payload in st.session_state.history:
        with st.chat_message(role):
            render_answer(payload) if role == "assistant" else st.markdown(payload)

    question = st.chat_input("輸入問題，例如：家人住院需要照顧，可以請什麼假？") or clicked
    if question:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("查詢條文並整理回答中…"):
                try:
                    ans = qa.ask(conn, question)
                except llm.NoProviderError as e:
                    ans = qa.Answer(f"⚠️ 目前無法連線到 LLM 服務，請稍後再試。（{e}）", [], None)
            render_answer(ans)
        st.session_state.history.append(("assistant", ans))

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
