"""資料來源：全國法規資料庫（逐條爬取）與公司工作規則（Markdown 檔）。

兩種來源都轉成同一種格式：一條條文一筆 Article。
"""

import hashlib
import json
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


# ── 勞動部函釋 ─────────────────────────────────────────────
# 函釋是主管機關對法條的解釋，很多實務問題（例如月薪制的 1 日工資怎麼算）條文沒寫，答案在函釋裡

INTERP_CODE = "MOL_INTERP"
INTERP_URL = "https://laws.mol.gov.tw/FLAW/FLAWDOC03.aspx?datatype=etype&N2={n2}&cnt=1&now=1&lnabndn=1&recordno=1"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_interpretation(n2: str, topic: str, html: str) -> Article:
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

    def between(start: str, end: str) -> str:
        m = re.search(start + r"\s*(.*?)\s*" + end, text, re.S)
        return _norm(m.group(1)) if m else ""

    doc_no = between("發文字號：", "發文日期：").replace(" ", "")
    date = between("發文日期：", "資料來源：")
    laws = between("相關法條：", "要　　旨：")
    gist = between("要　　旨：", r"(?:主\s*旨：|全文內容：)")
    m = re.search(r"((?:主\s*旨：|全文內容：).*?)\s*共\s*\d+\s*筆", text, re.S)
    body = _norm(m.group(1)) if m else ""
    # 正本、副本是發文對象，對回答沒有幫助
    body = re.split(r"\s正\s*本：", body)[0]
    if not doc_no or not (gist or body):
        raise ValueError(f"函釋 {n2} 解析失敗，網頁結構可能已改變")
    content = f"要旨：{gist}\n相關法條：{laws}\n{body}"
    return Article(n2, content, chapter=f"{date}｜{topic}", title=doc_no, url=INTERP_URL.format(n2=n2))


def interpretations_version(path: Path = config.INTERPRETATIONS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def fetch_interpretations(path: Path = config.INTERPRETATIONS_PATH) -> Source:
    """依精選清單逐則抓取原文。函釋發布後不會修改，所以版本以「清單內容」計算：清單沒變就不重抓。"""
    items = json.loads(path.read_text(encoding="utf-8"))
    articles = []
    with httpx.Client(headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT,
                      follow_redirects=True) as client:
        for it in items:
            resp = client.get(INTERP_URL.format(n2=it["n2"]))
            resp.raise_for_status()
            articles.append(parse_interpretation(it["n2"], it["topic"], resp.text))
            time.sleep(config.POLITE_DELAY)
    return Source(INTERP_CODE, "勞動部函釋（精選）", "interpretation", interpretations_version(path),
                  "https://laws.mol.gov.tw/", articles)


# ── 主管機關說明 ───────────────────────────────────────────
# 勞動局、勞動部發布的實務說明文章。有些實務規則（例如破月薪資的在職天數怎麼算）
# 不在法條也不在函釋裡，而是寫在這類說明中

GUIDANCE_CODE = "GUIDANCE"
SECTION_RE = re.compile(r"^[一二三四五六七八九十]+、")


def parse_guidance(item: dict, html: str) -> list[Article]:
    """從整頁擷取 item["title"] 到 item["end"] 之間的文章，依「一、二、三、」段落切成多塊。

    一頁電子報常有好幾篇文章，只取需要的那篇；文章較長，依段落切塊，檢索比較精準。
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    start = text.find(item["title"])
    end = text.find(item["end"], start + 1) if item.get("end") else -1
    if start < 0:
        raise ValueError(f"說明文章 {item['id']} 找不到標題，網頁結構可能已改變")
    lines = text[start + len(item["title"]): end if end > 0 else None].strip().splitlines()

    sections: list[tuple[str, list[str]]] = [("前言", [])]
    for line in lines:
        if SECTION_RE.match(line):
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    meta = f"{item['publisher']} {item['date']}"
    return [
        Article(f"{item['id']}-{i}", "\n".join(body), chapter=f"{meta}｜{heading}", title=item["title"], url=item["url"])
        for i, (heading, body) in enumerate(sections, start=1)
        if body
    ]


def fetch_guidance(path: Path = config.GUIDANCE_PATH) -> Source:
    items = json.loads(path.read_text(encoding="utf-8"))
    articles = []
    with httpx.Client(headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT,
                      follow_redirects=True) as client:
        for it in items:
            resp = client.get(it["url"])
            resp.raise_for_status()
            articles += parse_guidance(it, resp.text)
            time.sleep(config.POLITE_DELAY)
    return Source(GUIDANCE_CODE, "主管機關說明（精選）", "guidance", file_version(path), None, articles)


def file_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def source_version(conn: sqlite3.Connection, code: str) -> str | None:
    row = conn.execute("SELECT version FROM sources WHERE code = ?", (code,)).fetchone()
    return row["version"] if row else None


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
