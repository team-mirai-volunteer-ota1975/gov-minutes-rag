import re
import os
import sys
import time
import math
import uuid
import json
import logging
from typing import List, Optional, Iterable, Tuple

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
# Embedding providers
# -----------------------------

EMBED_DIM_DEFAULT = int(os.getenv("EMBED_DIM", "1536"))


class EmbeddingProvider:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbedding")
        try:
            import openai  # type: ignore
            self._client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            # Fallback for openai>=1.0 naming
            try:
                from openai import OpenAI  # type: ignore
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                raise RuntimeError("openai package not installed or incompatible") from e

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []

        # Exponential backoff with basic chunking
        i = 0
        while i < len(texts):
            chunk = texts[i:i + 100]  # batch
            attempt = 0
            while True:
                try:
                    resp = self._client.embeddings.create(model=self.model, input=chunk)
                    vecs = [item.embedding for item in resp.data]
                    out.extend(vecs)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > 5:
                        raise
                    backoff = min(60, 2 ** attempt)
                    logger.warning(f"OpenAI embed retry {attempt}, backing off {backoff}s: {e}")
                    time.sleep(backoff)
            i += len(chunk)
        return out


class LocalEmbedding(EmbeddingProvider):
    def __init__(self, model: Optional[str] = None, dim: int = EMBED_DIM_DEFAULT):
        self.model = model or os.getenv("EMBEDDING_MODEL", "local-hash")
        self.dim = dim
        # Try to use sentence-transformers if available; otherwise deterministic hash fallback
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            name = self.model or "intfloat/multilingual-e5-small"
            self._st_model = SentenceTransformer(name)
            # Infer dimension from model if possible
            try:
                sample = self._st_model.encode(["test"])
                if hasattr(sample, "shape") and len(sample.shape) == 2:
                    self.dim = int(sample.shape[1])
            except Exception:
                pass
            logger.info(f"Loaded local embedding model: {name}, dim={self.dim}")
        except Exception:
            logger.info("sentence-transformers not available; using hash-based local embeddings")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._st_model is not None:
            try:
                import numpy as np  # type: ignore
                vecs = self._st_model.encode(texts, normalize_embeddings=True)
                return [v.tolist() for v in vecs]
            except Exception as e:
                logger.warning(f"Local model failed, falling back to hash embeddings: {e}")

        # Deterministic hash-based embeddings (cosine-friendly)
        out: List[List[float]] = []
        for t in texts:
            seed = abs(hash(t))
            import random
            rnd = random.Random(seed)
            vec = [rnd.uniform(-1.0, 1.0) for _ in range(self.dim)]
            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            out.append(vec)
        return out


def get_embedding_provider_from_env() -> EmbeddingProvider:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if provider == "openai":
        return OpenAIEmbedding(model=model)
    if provider == "local":
        return LocalEmbedding(model=model)
    raise RuntimeError(f"Unknown EMBEDDING_PROVIDER: {provider}")


# -----------------------------
# Chunking
# -----------------------------

def split_into_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re_split_keep("\n\n+", text)]
    # If regex utility fails, fallback to simple split
    if not parts:
        parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def re_split_keep(pattern: str, s: str) -> List[str]:
    import re
    out: List[str] = []
    last = 0
    for m in re.finditer(pattern, s):
        out.append(s[last:m.start()])
        last = m.end()
    out.append(s[last:])
    return out


def chunk_text(text: str, target_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    # Approximation: 300-500 tokens ~ 1200-2000 chars
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= target_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur.strip())
            # if paragraph itself too long, slice it
            if len(p) > target_chars * 1.5:
                start = 0
                while start < len(p):
                    end = min(len(p), start + target_chars)
                    seg = p[start:end]
                    chunks.append(seg.strip())
                    start = max(end - overlap_chars, start + 1)
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur.strip())

    # Add overlaps between chunks
    with_overlaps: List[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            with_overlaps.append(c)
            continue
        prev = chunks[i - 1]
        tail = prev[-overlap_chars:]
        merged = (tail + "\n\n" + c).strip()
        with_overlaps.append(merged)
    return with_overlaps


# -----------------------------
# DB helpers
# -----------------------------

def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        logger.error("DATABASE_URL is not set in environment.")
        sys.exit(1)
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine


def vector_literal(vec: List[float]) -> str:
    # pgvector text literal format: '[v1, v2, ...]'
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"


def fetch_unembedded_docs(engine: Engine, limit: int = 100) -> List[Tuple[str, str]]:
    sql = text(
        """
        SELECT m.doc_id::text, m.discussion_text
        FROM meeting_metadata m
        LEFT JOIN meeting_chunks c ON c.doc_id = m.doc_id
        WHERE c.doc_id IS NULL AND m.discussion_text IS NOT NULL AND length(m.discussion_text) > 0
        LIMIT :limit
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(sql, {"limit": limit}).fetchall()
    return [(r[0], r[1]) for r in rows]


def delete_existing_chunks(engine: Engine, doc_id: uuid.UUID) -> None:
    sql = text("DELETE FROM meeting_chunks WHERE doc_id = :doc_id")
    with engine.begin() as conn:
        conn.execute(sql, {"doc_id": str(doc_id)})


def insert_chunks(engine: Engine, doc_id: uuid.UUID, chunks: List[str], embeddings: List[List[float]]) -> int:
    assert len(chunks) == len(embeddings)
    sql = text(
        """
        INSERT INTO meeting_chunks (doc_id, chunk_text, embedding)
        VALUES (:doc_id, :chunk_text, CAST(:embedding AS vector))
        """
    )
    count = 0
    with engine.begin() as conn:
        for chunk, vec in zip(chunks, embeddings):
            conn.execute(sql, {
                "doc_id": str(doc_id),
                "chunk_text": chunk,
                "embedding": vector_literal(vec),
            })
            count += 1
    return count


def embed_pending(limit_docs: int = 50):
    engine = get_engine()
    provider = get_embedding_provider_from_env()

    docs = fetch_unembedded_docs(engine, limit=limit_docs)
    if not docs:
        logger.info("No documents pending embedding.")
        return

    total_chunks = 0
    total_docs = 0

    for doc_id_str, text_content in docs:
        try:
            doc_id = uuid.UUID(doc_id_str)
            chunks = chunk_text(text_content)
            if not chunks:
                logger.info(f"No chunks for doc_id={doc_id}")
                continue
            # Embed with backoff is handled in provider
            embeddings = provider.embed(chunks)
            # Recreate chunks idempotently
            delete_existing_chunks(engine, doc_id)
            n = insert_chunks(engine, doc_id, chunks, embeddings)
            total_chunks += n
            total_docs += 1
            logger.info(f"Embedded doc_id={doc_id} chunks={n}")
        except Exception as e:
            logger.exception(f"Failed embedding doc {doc_id_str}: {e}")

    logger.info(f"Done. docs={total_docs} chunks={total_chunks}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chunk and embed meeting_metadata into meeting_chunks")
    parser.add_argument("--limit-docs", type=int, default=50)
    args = parser.parse_args()

    embed_pending(limit_docs=args.limit_docs)


if __name__ == "__main__":
    main()
