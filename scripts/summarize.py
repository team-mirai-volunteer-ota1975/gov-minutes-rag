import os
import re
import json
import time
import uuid
import math
import queue
import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

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
# Environment
# -----------------------------

SUMMARIZE_MODEL = os.getenv("SUMMARIZE_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-small"))
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "2000"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "3"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "60"))


# -----------------------------
# DB helpers
# -----------------------------

def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(url, pool_pre_ping=True, future=True)


def vector_literal(vec: List[float]) -> str:
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"


def select_candidates(engine: Engine, limit: int = 100) -> List[Tuple[str, Optional[str], str]]:
    sql = text(
        """
        SELECT m.doc_id::text, m.url, m.discussion_text
        FROM meeting_metadata m
        LEFT JOIN chunks_summary s USING (doc_id)
        WHERE s.doc_id IS NULL
          AND m.discussion_text IS NOT NULL
          AND length(m.discussion_text) > 0
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"limit": limit}).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def upsert_summary(
    engine: Engine,
    doc_id: uuid.UUID,
    summary: str,
    topics: List[str],
    model: str,
    embed_model: str,
    embedding: List[float],
) -> None:
    sql = text(
        """
        INSERT INTO chunks_summary (doc_id, summary, topics, model, embed_model, embedding)
        VALUES (:doc_id, :summary, :topics, :model, :embed_model, CAST(:embedding AS vector))
        ON CONFLICT (doc_id) DO UPDATE
        SET summary = EXCLUDED.summary,
            topics = EXCLUDED.topics,
            model = EXCLUDED.model,
            embed_model = EXCLUDED.embed_model,
            embedding = EXCLUDED.embedding,
            created_at = now()
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "doc_id": str(doc_id),
                "summary": summary,
                "topics": topics,
                "model": model,
                "embed_model": embed_model,
                "embedding": vector_literal(embedding),
            },
        )


# Optional jobs table support (ignore if absent)
def mark_status(engine: Engine, doc_id: uuid.UUID, status: str, last_error: Optional[str] = None) -> None:
    try:
        sql = text(
            """
            INSERT INTO chunks_summary_jobs (doc_id, status, last_error, updated_at)
            VALUES (:doc_id, :status, :last_error, now())
            ON CONFLICT (doc_id) DO UPDATE
            SET status = EXCLUDED.status,
                last_error = EXCLUDED.last_error,
                updated_at = now()
            """
        )
        with engine.begin() as conn:
            conn.execute(sql, {"doc_id": str(doc_id), "status": status, "last_error": last_error})
    except Exception:
        # Table not present; ignore
        pass


# -----------------------------
# Normalization & Chunking
# -----------------------------

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def normalize_text(s: str) -> str:
    t = nfkc(s)
    # Collapse spaces within lines
    lines = []
    for line in t.splitlines():
        line = line.rstrip()
        # Collapse internal whitespace to a single space
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    t = "\n".join(lines)
    # Keep at most 2 consecutive newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_text_chars(text: str, size: int, overlap: int, min_size: int = 1500) -> List[str]:
    if size <= 0:
        return [text.strip()]

    if len(text) <= min_size:
        return [text.strip()]

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        window = text[i:end]
        cut = None
        idx = window.rfind("。")
        if idx != -1 and (end - (i + idx + 1)) <= 200:
            cut = i + idx + 1
        j = cut or end
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        # 👇 min_size未満のchunkは overlapを無視して前進させる
        step = j - overlap if len(chunk) >= min_size else j
        i = max(step, i + 1)
    return chunks


# -----------------------------
# OpenAI client (chat + embeddings)
# -----------------------------


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.timeout = timeout
        try:
            import openai  # type: ignore
            self._client = openai.OpenAI(api_key=self.api_key, timeout=timeout)
            self._use_responses = False
        except Exception:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key, timeout=timeout)
            self._use_responses = False

    def chat_json(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        # Use Chat Completions with JSON schema-like output via response_format
        resp = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content or ""
        return content

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        i = 0
        while i < len(texts):
            batch = texts[i:i + 100]
            attempt = 0
            while True:
                try:
                    resp = self._client.embeddings.create(model=model, input=batch)
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > RETRY_MAX:
                        raise
                    backoff = min(60, 2 ** attempt)
                    logger.warning(f"embed retry {attempt} backoff {backoff}s: {e}")
                    time.sleep(backoff)
            i += len(batch)
        return out


# -----------------------------
# LLM Orchestration
# -----------------------------


MAP_PROMPT_TEMPLATE = (
    "あなたは政策担当者向けの議事録要約エージェントです。\n"
    "以下のテキスト断片を日本語で200〜300字に要約してください。\n"
    "必ず次を含めてください：\n"
    "- 目的/背景\n"
    "- 主要な論点（箇条書き可）\n"
    "- 結論/方向性 または 宿題（あれば）\n"
    "数値・日付・固有名詞は正確に保持し、推測しないこと。\n"
    "出力はJSONのみ：\n"
    '{{"summary":"...200〜300字..."}}'
    "\nテキスト:\n<<<\n{chunk}\n>>>"
)


def call_with_retry(fn: Callable[[], str], label: str) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= RETRY_MAX:
                break
            backoff = min(60, 2 ** attempt)
            logger.warning(f"{label} retry {attempt}/{RETRY_MAX}, backoff {backoff}s: {e}")
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def parse_json_strict(s: str) -> Dict[str, Any]:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be object")
    return obj


def llm_map_chunk(client: OpenAIClient, model: str, chunk: str) -> str:
    base_messages = [
        {"role": "system", "content": "JSONのみで出力し、スキーマを厳守してください。"},
        {"role": "user", "content": MAP_PROMPT_TEMPLATE.format(chunk=chunk)},
    ]

    def once(prompt_messages: List[Dict[str, str]]) -> str:
        content = client.chat_json(model=model, messages=prompt_messages, temperature=0.1)
        obj = parse_json_strict(content)
        summary = obj.get("summary")
        if not isinstance(summary, str):
            raise ValueError("summary missing or not a string")
        return summary.strip()

    # retry loop for JSON or API errors with one extra nudge
    try:
        return call_with_retry(lambda: once(base_messages), label="map")
    except Exception:
        nudged = base_messages + [
            {"role": "system", "content": "出力が壊れています。JSONスキーマ順守で再出力してください。"}
        ]
        return call_with_retry(lambda: once(nudged), label="map-nudge")


def hierarchical_reduce(client: OpenAIClient, model: str, partials: List[str]) -> Dict[str, Any]:
    def reduce_once(items: List[str]) -> Dict[str, Any]:
        bullet_list = "\n".join(f"- {s}" for s in items)
        prompt = (
            "以下は同一会議の部分要約一覧です。重複を排除し、矛盾がある場合は保守的に統一してください。\n"
            "政策担当者向けに、論点と結論/方向性が分かる要約（1000字程度）を作成し、\n"
            "主要トピック（名詞中心）を3〜8語抽出してください。\n"
            "出力は次のJSONのみ：\n"
            "{\n  \"summary\": \"政策担当者向け1000字要約\",\n  \"topics\": [\"...\", \"...\"]\n}\n"
            "部分要約一覧:\n"
            f"{bullet_list}"
        )
        messages = [
            {"role": "system", "content": "JSONのみで出力。スキーマ厳守。重複排除・保守的統一。"},
            {"role": "user", "content": prompt},
        ]

        def do() -> str:
            return client.chat_json(model=model, messages=messages, temperature=0.1)

        raw = call_with_retry(do, label="reduce")
        obj = parse_json_strict(raw)
        if not isinstance(obj.get("summary"), str) or not isinstance(obj.get("topics"), list):
            raise ValueError("invalid reduce JSON")
        return obj

    # If too many partials, reduce in groups then reduce again
    if len(partials) <= 20:
        return reduce_once(partials)

    # chunk into groups of 10
    mids: List[str] = []
    for i in range(0, len(partials), 10):
        group = partials[i:i + 10]
        mid = reduce_once(group)
        mids.append(mid.get("summary", ""))
    return reduce_once(mids)


def validate_reduce(obj: Dict[str, Any]) -> Dict[str, Any]:
    summary = (obj.get("summary") or "").strip()
    topics = obj.get("topics") or []
    if not isinstance(summary, str):
        raise ValueError("summary missing")
    if not isinstance(topics, list):
        raise ValueError("topics missing")

    # enforce 100-1500 chars
    n = len(summary)
    ok_len = 100 <= n <= 1500
    # normalize topics: nfkc + lower + dedupe + max len 20 chars, keep 3-8
    norm_topics: List[str] = []
    seen = set()
    for t in topics:
        if not isinstance(t, str):
            continue
        tt = unicodedata.normalize("NFKC", t).lower().strip()
        if not tt or len(tt) > 20:
            tt = tt[:20]
        if tt and tt not in seen:
            seen.add(tt)
            norm_topics.append(tt)
    ok_topics = 3 <= len(norm_topics) <= 8

    if not (ok_len and ok_topics):
        raise ValueError(f"validation failed len={n} topics={len(norm_topics)}")
    return {"summary": summary, "topics": norm_topics}


# -----------------------------
# Main pipeline
# -----------------------------


def run_for_doc(
    engine: Engine,
    client: OpenAIClient,
    doc_id_str: str,
    url: Optional[str],
    discussion_text: str,
) -> None:
    doc_id = uuid.UUID(doc_id_str)
    mark_status(engine, doc_id, "running")

    t0 = time.time()
    text = normalize_text(discussion_text)
    chunks = split_text_chars(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    # 20文字未満は除外
    chunks = [c for c in chunks if len(c) >= 20]
    if not chunks:
        raise ValueError("no chunks from discussion_text")

    # Map step with simple bounded concurrency (threads)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def map_one(c: str) -> str:
        return llm_map_chunk(client, SUMMARIZE_MODEL, c)

    map_summaries: List[str] = []
    with ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENCY)) as ex:
        futures = [ex.submit(map_one, c) for c in chunks]
        for fut in as_completed(futures):
            map_summaries.append(fut.result())

    # Reduce step with one validation retry if needed
    reduced = hierarchical_reduce(client, SUMMARIZE_MODEL, map_summaries)
    try:
        reduced_ok = validate_reduce(reduced)
    except Exception:
        # one more strongly worded pass
        note = (
            "以下の部分要約で再度要約してください。500〜800字を厳守し、topicsは重複排除で3〜8語。"
        )
        second = hierarchical_reduce(client, SUMMARIZE_MODEL, [note] + map_summaries)
        reduced_ok = validate_reduce(second)

    # Embed summary
    emb = client.embed(EMBED_MODEL, [reduced_ok["summary"]])[0]

    # Upsert
    upsert_summary(
        engine,
        doc_id=doc_id,
        summary=reduced_ok["summary"],
        topics=reduced_ok["topics"],
        model=SUMMARIZE_MODEL,
        embed_model=EMBED_MODEL,
        embedding=emb,
    )

    mark_status(engine, doc_id, "done")
    dt = time.time() - t0
    logger.info(f"doc {doc_id} summarized in {dt:.1f}s, chunks={len(chunks)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Map→Reduce summarize meeting_metadata into chunks_summary")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    engine = get_engine()
    client = OpenAIClient(timeout=OPENAI_TIMEOUT_SEC)

    rows = select_candidates(engine, limit=args.limit)
    if not rows:
        logger.info("No candidates found for chunks_summary.")
        return

    for doc_id, url, text in rows:
        try:
            run_for_doc(engine, client, doc_id, url, text)
        except Exception as e:
            try:
                mark_status(engine, uuid.UUID(doc_id), "error", str(e))
            except Exception:
                pass
            logger.exception(f"Failed doc_id={doc_id}: {e}")


if __name__ == "__main__":
    main()
