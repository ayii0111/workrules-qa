"""資料來源：全國法規資料庫（逐條爬取）與公司工作規則（Markdown 檔）。

兩種來源都轉成同一種格式：一條條文一筆 Article。
"""

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from . import config

LAW_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={code}"
ARTICLE_URL = "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={code}&flno={flno}"
HANDBOOK_CODE = "HANDBOOK"
FLNO_RE = re.compile(r"第\s*([\d\-]+)\s*條")


@dataclass
class Article:
    flno: str
    content: str
    chapter: str | None = None
    title: str | None = None
    url: str | None = None


@dataclass
class Source:
    code: str
    name: str
    kind: str
    version: str
    url: str | None
    articles: list[Article]


# ── 法規 ───────────────────────────────────────────────────

def parse_law(code: str, html: str) -> Source:
    soup = BeautifulSoup(html, "html.parser")
    name = soup.title.get_text(strip=True).split("-")[0] if soup.title else config.LAWS.get(code, code)

    version = ""
    for tr in soup.select("table tr"):
        th = tr.find("th")
        if th and ("修正日期" in th.get_text() or "公布日期" in th.get_text()):
            version = tr.get_text(" ", strip=True).split("：", 1)[-1].strip()
            if "修正日期" in th.get_text():
                break

    articles, chapter = [], None
    container = soup.select_one(".law-reg-content")
    for el in container.find_all("div", recursive=False) if container else []:
        classes = el.get("class") or []
        if "h3" in classes:
            chapter = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        elif "row" in classes:
            no = el.select_one(".col-no a")
            body = el.select_one(".law-article")
            if not no or not body:
                continue
            m = FLNO_RE.search(no.get_text())
            content = "\n".join(line.get_text(strip=True) for line in body.find_all("div")) or body.get_text(strip=True)
            if not m or content in ("（刪除）", "(刪除)"):
                continue
            flno = m.group(1)
            articles.append(Article(flno, content, chapter, url=ARTICLE_URL.format(code=code, flno=flno)))
    return Source(code, name, "law", version, LAW_URL.format(code=code), articles)


def fetch_laws(codes: list[str]) -> list[Source]:
    sources = []
    with httpx.Client(headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT,
                      follow_redirects=True) as client:
        for code in codes:
            resp = client.get(LAW_URL.format(code=code))
            resp.raise_for_status()
            src = parse_law(code, resp.text)
            if not src.articles:
                raise ValueError(f"{code} 解析不到任何條文，網頁結構可能已改變")
            sources.append(src)
            time.sleep(config.POLITE_DELAY)
    return sources


# ── 公司工作規則 ───────────────────────────────────────────

HEADING_RE = re.compile(r"^###\s*第\s*(\d+)\s*條\s*(.*)$")


def parse_handbook(text: str) -> Source:
    """格式：`# 名稱`、`## 章`、`### 第 N 條 標題`，其下為條文內容。"""
    name, chapter, articles, current = "公司工作規則", None, [], None
    for line in text.splitlines():
        if line.startswith("# "):
            name = line[2:].strip()
        elif line.startswith("## "):
            chapter = line[3:].strip()
        elif m := HEADING_RE.match(line):
            current = Article(m.group(1), "", chapter, m.group(2).strip())
            articles.append(current)
        elif current is not None and line.strip() and not line.startswith(">"):
            current.content += ("\n" if current.content else "") + line.strip()
    version = hashlib.sha256(text.encode()).hexdigest()[:12]
    return Source(HANDBOOK_CODE, name, "handbook", version, None, [a for a in articles if a.content])


def load_handbook(path: Path = config.HANDBOOK_PATH) -> Source:
    return parse_handbook(path.read_text(encoding="utf-8"))


# ── 寫入 ───────────────────────────────────────────────────

def save_source(conn: sqlite3.Connection, src: Source) -> int:
    """版本沒變就略過；有變就整部替換（法規修正時可能增刪條文，逐條比對容易出錯）。
    回傳寫入的條文數。"""
    row = conn.execute("SELECT version FROM sources WHERE code = ?", (src.code,)).fetchone()
    if row and row["version"] == src.version:
        return 0
    old_ids = [r["id"] for r in conn.execute("SELECT id FROM articles WHERE code = ?", (src.code,))]
    if old_ids:
        conn.execute(f"DELETE FROM articles_fts WHERE article_id IN ({','.join('?' * len(old_ids))})", old_ids)
    conn.execute("DELETE FROM articles WHERE code = ?", (src.code,))
    conn.execute(
        """INSERT INTO sources (code, name, kind, version, url) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(code) DO UPDATE SET name = excluded.name, version = excluded.version,
                                           url = excluded.url, fetched_at = datetime('now')""",
        (src.code, src.name, src.kind, src.version, src.url),
    )
    conn.executemany(
        "INSERT INTO articles (code, flno, title, chapter, content, url) VALUES (?, ?, ?, ?, ?, ?)",
        [(src.code, a.flno, a.title, a.chapter, a.content, a.url) for a in src.articles],
    )
    return len(src.articles)
