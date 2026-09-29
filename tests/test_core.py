from datetime import date
from fractions import Fraction

from workrules import db, index, sources
from workrules.calc import annual_leave_days, leave_schedule, months_between, overtime_pay
from workrules.qa import cited_hits
from workrules.retrieval import Hit, explicit_refs, keyword_search, rrf

LAW_HTML = """<html><head><title>勞動基準法-全國法規資料庫</title></head><body>
<table><tr><th>法規名稱：</th><td>勞動基準法</td></tr><tr><th>修正日期：</th><td>民國 113 年 07 月 31 日</td></tr></table>
<div class="law-reg-content">
  <div class="h3 char-2">第 四 章 工作時間、休息、休假</div>
  <div class="row"><div class="col-no"><a href="x">第 32-1 條</a></div>
    <div class="col-data"><div class="law-article"><div class="line-0000">雇主依第三十二條第一項及第二項規定使勞工延長工作時間…</div><div class="line-0000">前項之補休…</div></div></div></div>
  <div class="row"><div class="col-no"><a href="x">第 33 條</a></div>
    <div class="col-data"><div class="law-article"><div class="line-0000">（刪除）</div></div></div></div>
</div></body></html>"""

HANDBOOK_MD = """# 範例公司 員工工作規則
> 虛構範例
## 第二章 工作時間
### 第 7 條 補休
員工加班一律以補休方式處理。
"""


# ── 資料來源解析 ───────────────────────────────────────────

def test_parse_law_extracts_version_chapter_and_skips_deleted():
    src = sources.parse_law("N0030001", LAW_HTML)
    assert src.name == "勞動基準法"
    assert src.version == "民國 113 年 07 月 31 日"
    assert [a.flno for a in src.articles] == ["32-1"]
    art = src.articles[0]
    assert art.chapter == "第 四 章 工作時間、休息、休假"
    assert art.content.count("\n") == 1
    assert art.url.endswith("pcode=N0030001&flno=32-1")


def test_parse_handbook_skips_notes_and_keeps_titles():
    src = sources.parse_handbook(HANDBOOK_MD)
    assert src.name == "範例公司 員工工作規則"
    [a] = src.articles
    assert (a.flno, a.title, a.chapter) == ("7", "補休", "第二章 工作時間")
    assert "虛構" not in a.content


def test_save_source_skips_same_version_and_replaces_new_version(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    src = sources.parse_handbook(HANDBOOK_MD)
    assert sources.save_source(conn, src) == 1
    index.index_missing(conn)
    assert sources.save_source(conn, src) == 0
    changed = sources.parse_handbook(HANDBOOK_MD + "\n### 第 8 條 請假\n請假應事先申請。\n")
    assert sources.save_source(conn, changed) == 2
    # 舊版的全文索引要一起清掉，避免搜到已不存在的條文
    assert conn.execute("SELECT count(*) FROM articles_fts").fetchone()[0] == 0


# ── 檢索 ───────────────────────────────────────────────────

def _db_with_handbook_and_law(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    sources.save_source(conn, sources.parse_law("N0030001", LAW_HTML))
    sources.save_source(conn, sources.parse_handbook(HANDBOOK_MD))
    index.index_missing(conn)
    return conn


def test_keyword_search_finds_chinese_terms(tmp_path):
    conn = _db_with_handbook_and_law(tmp_path)
    ids = keyword_search(conn, "補休的規定")
    assert ids


def test_explicit_refs_match_law_name_and_alias(tmp_path):
    conn = _db_with_handbook_and_law(tmp_path)
    law_id = conn.execute("SELECT id FROM articles WHERE flno = '32-1'").fetchone()["id"]
    hb_id = conn.execute("SELECT id FROM articles WHERE code = 'HANDBOOK'").fetchone()["id"]
    assert explicit_refs(conn, "勞基法第 32-1 條在講什麼") == [law_id]
    assert explicit_refs(conn, "工作規則第7條") == [hb_id]
    assert explicit_refs(conn, "勞動基準法施行細則第 32-1 條") == []


def test_rrf_rewards_items_ranked_by_both_methods():
    scores = rrf([[1, 2, 3], [3, 4]])
    assert max(scores, key=scores.get) == 3


def test_cited_hits_keeps_only_cited_in_order():
    law = Hit(1, "N0030001", "勞動基準法", "law", "38", None, "", None, 1)
    hb = Hit(2, "HANDBOOK", "範例公司 員工工作規則", "handbook", "9", "特別休假", "", None, 1)
    other = Hit(3, "N0030006", "勞工請假規則", "law", "4", None, "", None, 1)
    text = "可放 7 天【工作規則 第 9 條、勞動基準法 第 38 條】"
    assert cited_hits(text, [law, other, hb]) == [hb, law]
    assert cited_hits("沒有引用", [law, other]) == [law, other]


# ── 試算 ───────────────────────────────────────────────────

def test_months_between_counts_full_months_only():
    assert months_between(date(2025, 1, 31), date(2025, 7, 30)) == 5
    assert months_between(date(2025, 1, 31), date(2025, 7, 31)) == 6


def test_annual_leave_days_follow_article_38():
    start = date(2020, 3, 1)
    cases = {
        date(2020, 8, 31): 0,   # 未滿 6 個月
        date(2020, 9, 1): 3,    # 6 個月以上 1 年未滿
        date(2021, 3, 1): 7,    # 1 年
        date(2022, 3, 1): 10,   # 2 年
        date(2024, 3, 1): 14,   # 4 年
        date(2025, 3, 1): 15,   # 5 年
        date(2030, 3, 1): 16,   # 10 年
        date(2045, 3, 1): 30,   # 25 年，上限 30 日
    }
    for on, days in cases.items():
        assert annual_leave_days(start, on) == days, on


def test_leave_schedule_handles_month_end():
    sched = leave_schedule(date(2025, 8, 31), years=1)
    assert sched[0] == (date(2026, 2, 28), 3)
    assert sched[1] == (date(2026, 8, 31), 7)


def test_overtime_pay_workday_tiers():
    # 月薪 36,000 → 時薪 150；3 小時 = 2×150×4/3 + 1×150×5/3 = 400 + 250
    lines = overtime_pay(36000, 3, "workday")
    assert [(l.hours, l.rate, l.amount) for l in lines] == [(2, Fraction(4, 3), 400), (1, Fraction(5, 3), 250)]


def test_overtime_pay_restday_tiers_and_limit():
    lines = overtime_pay(36000, 10, "restday")
    assert [l.amount for l in lines] == [400, 1500, 800]
    try:
        overtime_pay(36000, 5, "workday")
    except ValueError:
        pass
    else:
        raise AssertionError("平日加班超過 4 小時應該報錯")
