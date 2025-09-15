import os
import logging
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(url, pool_pre_ping=True, future=True)


# Try to reuse provider from scripts.embed; fallback to local implementation
def vector_literal(vec: list[float]) -> str:
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"


class SimpleLocalEmbed:
    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import math, random
        out = []
        for t in texts:
            rnd = random.Random(abs(hash(t)))
            vec = [rnd.uniform(-1.0, 1.0) for _ in range(self.dim)]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            out.append(vec)
        return out


def get_embedding_provider():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if provider == "openai":
        try:
            from scripts.embed import OpenAIEmbedding  # type: ignore
            return OpenAIEmbedding(model=model)
        except Exception as e:
            logger.warning(f"Falling back to local embeddings: {e}")
            return SimpleLocalEmbed()
    else:
        try:
            from scripts.embed import LocalEmbedding  # type: ignore
            return LocalEmbedding(model=model)
        except Exception:
            return SimpleLocalEmbed()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    ministry: Optional[str] = None


app = FastAPI(title="Gov Minutes RAG API")
engine = get_engine()
provider = get_embedding_provider()


@app.get("/healthz")
def healthz():
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest) -> List[Dict[str, Any]]:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    top_k = max(1, min(100, req.top_k or 5))

    # Embed query
    vec = provider.embed([req.query])[0]
    vec_lit = vector_literal(vec)

    sql = text(
        """
        SELECT m.url, m.council_name, m.date, c.chunk_text,
               1 - (c.embedding <=> :query_vec::vector) AS score
        FROM meeting_chunks c
        JOIN meeting_metadata m ON c.doc_id = m.doc_id
        WHERE (:ministry IS NULL OR m.ministry = :ministry)
        ORDER BY c.embedding <=> :query_vec::vector
        LIMIT :top_k
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, {
            "query_vec": vec_lit,
            "ministry": req.ministry,
            "top_k": top_k,
        }).fetchall()

    results = []
    for r in rows:
        results.append({
            "url": r[0],
            "council_name": r[1],
            "date": r[2].isoformat() if r[2] else None,
            "chunk_text": r[3],
            "score": float(r[4]) if r[4] is not None else None,
        })
    return results

