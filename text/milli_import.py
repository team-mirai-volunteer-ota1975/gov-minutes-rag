import os
import logging
import requests
import fugashi
from dotenv import load_dotenv
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

tagger = fugashi.Tagger()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MEILI_URL = os.getenv("MEILI_URL", "http://127.0.0.1:7700")
MEILI_KEY = os.getenv("MEILI_KEY", "MASTER_KEY")
MEILI_INDEX = os.getenv("MEILI_INDEX", "meetings")

def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(url, pool_pre_ping=True, future=True)

def fetch_records(engine: Engine, limit: int = 100) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT m.doc_id, m.url, s.summary, m.discussion_text
        FROM meeting_metadata m
        JOIN chunks_summary s ON m.doc_id = s.doc_id
        ORDER BY m.date DESC NULLS LAST
        LIMIT :limit
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().all()
        return [dict(r) for r in rows]

def push_to_meilisearch(docs: List[Dict[str, Any]]):
    url = f"{MEILI_URL}/indexes/{MEILI_INDEX}/documents"
    headers = {
        "Authorization": f"Bearer {MEILI_KEY}",
        "Content-Type": "application/json",
    }
    res = requests.post(url, headers=headers, json=docs)
    res.raise_for_status()
    logger.info(f"Pushed {len(docs)} docs to Meilisearch (updateId={res.json().get('updateId')})")

def tokenize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(word.surface for word in tagger(text))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export meeting data from Postgres to Meilisearch")
    parser.add_argument("--limit", type=int, default=100, help="取得件数")
    args = parser.parse_args()

    engine = get_engine()
    records = fetch_records(engine, args.limit)

    if not records:
        logger.info("No records found.")
        return

    # Meilisearch 用に整形（doc_id を id として使う）
    docs = []
    for r in records:
        docs.append({
            "id": str(r["doc_id"]),   # UUID を文字列に変換
            "url": r["url"],
            "summary": tokenize_text(r["summary"]),  # ← MeCabで形態素分割
            "discussion_text": tokenize_text(r["discussion_text"]),
        })

    push_to_meilisearch(docs)

if __name__ == "__main__":
    main()
