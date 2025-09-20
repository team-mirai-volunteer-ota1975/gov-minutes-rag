import os
import re
import sys
import json
import uuid
import logging
import unicodedata
from datetime import date
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------

JP_ERAS = {
    "令和": 2018,  # year + n -> western = 2018 + n (since 2019 is 1)
    "平成": 1988,
    "昭和": 1925,
    "大正": 1911,
    "明治": 1867,
}

DOMAIN_MINISTRY_MAP = {
    "cao.go.jp": "内閣府",
    "kantei.go.jp": "内閣府",
    "soumu.go.jp": "総務省",
    "moj.go.jp": "法務省",
    "mofa.go.jp": "外務省",
    "mof.go.jp": "財務省",
    "mext.go.jp": "文部科学省",
    "mhlw.go.jp": "厚生労働省",
    "maff.go.jp": "農林水産省",
    "meti.go.jp": "経済産業省",
    "mlit.go.jp": "国土交通省",
    "env.go.jp": "環境省",
    "mod.go.jp": "防衛省",
    "digital.go.jp": "デジタル庁",
}

def nfkc(s: str) -> str:
    if not s:
        return s
    return unicodedata.normalize("NFKC", s)


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = nfkc(s)
    # Remove excessive spaces, page numbers like "- 12 -", headers/footers heuristics
    lines = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue
        # Drop standalone page numbers
        if re.fullmatch(r"-?\s*\d+\s*-?", line):
            continue
        # Collapse internal spaces
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    # Collapse 3+ newlines to 2
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return out.strip()


def fullwidth_to_halfwidth_digits(s: str) -> str:
    return re.sub(
        r"[０-９]",
        lambda m: chr(ord(m.group(0)) - ord("０") + ord("0")),
        s,
    )


def parse_japanese_date(s: str) -> Optional[date]:
    if not s:
        return None
    t = nfkc(s)
    t = fullwidth_to_halfwidth_digits(t)

    # Era date: 令和7年7月23日 etc
    m = re.search(r"(令和|平成|昭和|大正|明治)\s*(\d+)年\s*(\d{1,2})月\s*(\d{1,2})日", t)
    if m:
        era, y, mo, d = m.groups()
        try:
            base = JP_ERAS[era]
            return date(base + int(y), int(mo), int(d))
        except Exception:
            pass

    # Western date: 2025年7月23日 or 2025/07/23 or 2025-07-23
    m = re.search(r"(20\d{2}|19\d{2})[年/\-.]\s*(\d{1,2})[月/\-.]\s*(\d{1,2})日?", t)
    if m:
        y, mo, d = m.groups()
        try:
            return date(int(y), int(mo), int(d))
        except Exception:
            pass

    return None


def guess_ministry_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    netloc = (parsed.netloc or "").lower()
    if not netloc:
        return None

    netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0]

    for domain, ministry in DOMAIN_MINISTRY_MAP.items():
        domain = domain.lower()
        if netloc == domain or netloc.endswith(f".{domain}"):
            return ministry
    return None


def guess_ministry(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = [
        "内閣府", "総務省", "法務省", "外務省", "財務省", "文部科学省",
        "厚生労働省", "農林水産省", "経済産業省", "国土交通省", "環境省",
        "防衛省", "デジタル庁"
    ]
    for c in candidates:
        if c in text:
            return c
    return None


def extract_title(text: str) -> Optional[str]:
    # Take first non-empty line with likely meeting words
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(k in line for k in ["審議会", "委員会", "会議", "部会", "検討会"]):
            return line[:200]
    # Fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return None


def extract_meeting_no(text: str) -> Optional[str]:
    # e.g., 第12回, 第１２回, 第百三回
    m = re.search(r"第\s*([0-9０-９一二三四五六七八九十百千]+)\s*回", text)
    if m:
        return nfkc(m.group(0))
    return None


def extract_location(text: str) -> Optional[str]:
    # Look for lines with 場所 or 会場
    for line in text.splitlines():
        if "場所" in line or "会場" in line:
            seg = re.split(r"[:：]", line, maxsplit=1)
            if len(seg) == 2:
                return seg[1].strip()[:200]
            return line.strip()[:200]
    return None


def extract_attendees(text: str) -> List[str]:
    # Heuristic: lines after 出席者 or 出席委員
    m = re.search(r"(出席者|出席委員)[：:\n]\s*(.+)", text)
    if m:
        names = re.split(r"[、,\s]", m.group(2))
        names = [n.strip() for n in names if n.strip()]
        return names[:100]
    return []


def extract_agenda(text: str) -> str:
    if not text:
        return ""

    # 開始位置を検索（議題 or 議事）
    m = re.search(r"(議題|議事(?!要旨|概要|録))", text)
    if not m:
        return ""

    start = m.start()

    # そこから 200文字を切り出し
    snippet = text[start:start + 200]

    # 終了条件（出席者などの見出しが出たらそこで切る）
    stop_keywords = ["出席者", "配布資料", "議事", "議事内容", "開会"]
    stop_positions = [snippet.find(k) for k in stop_keywords if snippet.find(k) != -1]
    if stop_positions:
        end = min(stop_positions)
        snippet = snippet[:end]

    # 改行をスペースにリプレース
    snippet = snippet.replace("\n", " ")

    return snippet.strip()


# -----------------------------
# Content extraction
# -----------------------------

def extract_text_from_html_bytes(b: bytes) -> str:
    text = ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(b, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        body_text = soup.get_text("\n")
        text = f"{title}\n{body_text}" if title else body_text
    except Exception:
        # Fallback: crude strip tags
        s = b.decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", "\n", s)
    return clean_text(text)


def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=b, filetype="pdf")
        pages = []
        for page in doc:
            try:
                # "text" モードなら段落っぽく整形してくれる
                text = page.get_text("text")
                pages.append(text or "")
            except Exception:
                continue
        return clean_text("\n\n".join(pages))
    except Exception as e:
        logger.warning(f"PDF parsing failed with PyMuPDF: {e}")
        return ""


def extract_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            b = f.read()
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return ""

    if ext in [".html", ".htm"]:
        return extract_text_from_html_bytes(b)
    if ext in [".pdf"]:
        return extract_text_from_pdf_bytes(b)
    # Plain text
    try:
        return clean_text(b.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


# -----------------------------
# DB interactions
# -----------------------------

def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL is not set in environment.")
        sys.exit(1)
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine


def upsert_meeting(engine: Engine, url: str, text_content: str, html_title: Optional[str] = None) -> Optional[uuid.UUID]:
    if not url:
        logger.warning("Skipping record with empty URL")
        return None

    discussion = clean_text(text_content)
    title_line = extract_title((html_title or "") + "\n" + discussion)
    ministry = guess_ministry_from_url(url) or guess_ministry((html_title or "") + "\n" + discussion)
    meeting_no = extract_meeting_no(discussion)
    dt = parse_japanese_date((html_title or "") + "\n" + discussion)
    location = extract_location(discussion)
    attendees = extract_attendees(discussion)
    agenda = extract_agenda(discussion)

    # Stable UUID based on URL to be idempotent
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, nfkc(url))

    stmt = text(
        """
        INSERT INTO meeting_metadata (doc_id, url, ministry, council_name, meeting_no, date, location, attendees, agenda, discussion_text)
        VALUES (:doc_id, :url, :ministry, :council_name, :meeting_no, :date, :location, CAST(:attendees AS JSONB), CAST(:agenda AS JSONB), :discussion_text)
        ON CONFLICT (url) DO UPDATE SET
            ministry = EXCLUDED.ministry,
            council_name = EXCLUDED.council_name,
            meeting_no = EXCLUDED.meeting_no,
            date = EXCLUDED.date,
            location = EXCLUDED.location,
            attendees = EXCLUDED.attendees,
            agenda = EXCLUDED.agenda,
            discussion_text = EXCLUDED.discussion_text
        RETURNING doc_id
        """
    )

    params = {
        "doc_id": str(doc_id),
        "url": url,
        "ministry": ministry,
        "council_name": title_line,
        "meeting_no": meeting_no,
        "date": dt,
        "location": location,
        "attendees": json.dumps(attendees or []),
        "agenda": json.dumps(agenda or []),
        "discussion_text": discussion,
    }

    with engine.begin() as conn:
        res = conn.execute(stmt, params)
        row = res.first()
        if row:
            return uuid.UUID(str(row[0]))
    return None


def process_crawled_pages(engine: Engine, limit: int = 100) -> Tuple[int, int]:
    success = 0
    failed = 0
    # Select pages not yet in meeting_metadata by URL
    select_sql = text(
        """
        SELECT p.url, p.html_title, p.content_type, p.content
        FROM crawled_pages p
        LEFT JOIN meeting_metadata m ON m.url = p.url
        WHERE m.url IS NULL
        ORDER BY p.fetched_at DESC
        LIMIT :limit
        """
    )
    try:
        with engine.begin() as conn:
            rows = conn.execute(select_sql, {"limit": limit}).fetchall()
    except Exception as e:
        logger.warning(f"Failed to query crawled_pages (maybe missing?): {e}")
        return (0, 0)

    for url, html_title, content_type, content in rows:
        try:
            text_content = ""
            if content:
                b = bytes(content)
                if (content_type or "").lower().startswith("text/html"):
                    text_content = extract_text_from_html_bytes(b)
                elif (content_type or "").lower().startswith("application/pdf"):
                    text_content = extract_text_from_pdf_bytes(b)
                else:
                    # Try both parsers heuristically
                    text_content = extract_text_from_html_bytes(b)
                    if len(text_content) < 50:
                        text_content = extract_text_from_pdf_bytes(b)
            doc_id = upsert_meeting(engine, url=url, text_content=text_content, html_title=html_title)
            if doc_id:
                success += 1
        except Exception as e:
            logger.exception(f"Failed processing URL={url}: {e}")
            failed += 1

    return success, failed


def process_local_samples(engine: Engine, sample_dir: str = os.path.join(os.path.dirname(__file__), "..", "sample")) -> Tuple[int, int]:
    success = 0
    failed = 0
    path = os.path.abspath(sample_dir)
    if not os.path.isdir(path):
        logger.warning(f"Sample dir not found: {path}")
        return (0, 0)

    for root, _, files in os.walk(path):
        for fn in files:
            if not fn.lower().endswith((".pdf", ".html", ".htm", ".txt")):
                continue
            fpath = os.path.join(root, fn)
            try:
                text_content = extract_text_from_path(fpath)
                pseudo_url = f"file://{os.path.relpath(fpath, path).replace('\\', '/')}"
                doc_id = upsert_meeting(engine, url=pseudo_url, text_content=text_content, html_title=None)
                if doc_id:
                    success += 1
            except Exception as e:
                logger.exception(f"Failed processing sample={fpath}: {e}")
                failed += 1

    return success, failed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Normalize crawled pages and local samples into meeting_metadata")
    parser.add_argument("--source", choices=["db", "sample", "both"], default="both")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-dir", default=os.path.join(os.path.dirname(__file__), "..", "sample"))
    args = parser.parse_args()

    engine = get_engine()

    total_ok = 0
    total_ng = 0

    if args.source in ("db", "both"):
        ok, ng = process_crawled_pages(engine, limit=args.limit)
        logger.info(f"DB processed: ok={ok} ng={ng}")
        total_ok += ok
        total_ng += ng

    if args.source in ("sample", "both"):
        ok, ng = process_local_samples(engine, sample_dir=args.sample_dir)
        logger.info(f"Samples processed: ok={ok} ng={ng}")
        total_ok += ok
        total_ng += ng

    logger.info(f"Done. Total ok={total_ok} ng={total_ng}")


if __name__ == "__main__":
    main()
