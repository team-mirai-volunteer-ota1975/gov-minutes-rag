import os
import logging
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
# from rank_bm25 import BM25Okapi
# from fugashi import Tagger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# tagger = Tagger()


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
               1 - (c.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM meeting_chunks c
        JOIN meeting_metadata m ON c.doc_id = m.doc_id
        WHERE (:ministry IS NULL OR m.ministry = :ministry)
        ORDER BY c.embedding <=> CAST(:query_vec AS vector)
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


@app.post("/summary_search")
def summary_search(req: SearchRequest) -> List[Dict[str, Any]]:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    top_k = max(1, min(100, req.top_k or 5))

    try:
        vec = provider.embed([req.query])[0]
    except Exception as e:
        logger.exception(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
    vec_lit = vector_literal(vec)

    sql = text(
        """
        SELECT m.url, m.council_name, m.date, s.summary,
               1 - (s.embedding <=> CAST(:query_vec AS vector)) AS score
        FROM chunks_summary s
        JOIN meeting_metadata m ON s.doc_id = m.doc_id
        WHERE (:ministry IS NULL OR m.ministry = :ministry)
        ORDER BY s.embedding <=> CAST(:query_vec AS vector)
        LIMIT :top_k
        """
    )

    try:
        with engine.begin() as conn:
            rows = conn.execute(
                sql,
                {
                    "query_vec": vec_lit,
                    "ministry": req.ministry,
                    "top_k": top_k,
                },
            ).fetchall()
    except Exception as e:
        logger.exception(f"DB query failed: {e}")
        raise HTTPException(status_code=500, detail=f"db query failed: {e}")

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "url": r[0],
                "council_name": r[1],
                "date": r[2].isoformat() if r[2] else None,
                "summary": r[3],
                "score": float(r[4]) if r[4] is not None else None,
            }
        )
    return results


# class CompareRequest(BaseModel):
#     query: str
#     top_k: int = 5


# def tokenize(text: str) -> List[str]:
#     """fugashiで日本語分かち書き"""
#     return [word.surface for word in tagger(text)]


# @app.post("/debug/compare_search")
# def compare_search(req: CompareRequest) -> Dict[str, Any]:
#     if not req.query or not req.query.strip():
#         raise HTTPException(status_code=400, detail="query is required")
#     top_k = max(1, min(100, req.top_k or 5))

#     # Embed query
#     try:
#         vec = provider.embed([req.query])[0]
#     except Exception as e:
#         logger.exception(f"Embedding failed: {e}")
#         raise HTTPException(status_code=500, detail=f"embedding failed: {e}")
#     vec_lit = vector_literal(vec)

#     sql_chunks = text(
#         """
#         SELECT c.doc_id::text, m.url, c.chunk_text,
#             1 - (c.embedding <=> CAST(:query_vec AS vector)) AS score
#         FROM meeting_chunks c
#         JOIN meeting_metadata m USING (doc_id)
#         ORDER BY c.embedding <=> CAST(:query_vec AS vector)
#         LIMIT :top_k
#         """
#     )

#     sql_summaries = text(
#         """
#         SELECT s.doc_id::text, m.url, s.summary,
#             1 - (s.embedding <=> CAST(:query_vec AS vector)) AS score
#         FROM chunks_summary s
#         JOIN meeting_metadata m USING (doc_id)
#         ORDER BY s.embedding <=> CAST(:query_vec AS vector)
#         LIMIT :top_k
#         """
#     )

#     try:
#         with engine.begin() as conn:
#             rows_chunks = conn.execute(sql_chunks, {"query_vec": vec_lit, "top_k": top_k}).fetchall()
#             rows_summ = conn.execute(sql_summaries, {"query_vec": vec_lit, "top_k": top_k}).fetchall()
#     except Exception as e:
#         logger.exception(f"DB query failed: {e}")
#         raise HTTPException(status_code=500, detail=f"db query failed: {e}")

#     # --- Chunks 出力 ---
#     chunks_out: List[Dict[str, Any]] = []
#     for r in rows_chunks:
#         chunks_out.append({
#             "doc_id": r[0],
#             "url": r[1],
#             "text": r[2],
#             "score": float(r[3]) if r[3] is not None else None,
#         })

#     # --- Summaries 出力 ---
#     summaries_out: List[Dict[str, Any]] = []
#     for r in rows_summ:
#         summaries_out.append({
#             "doc_id": r[0],
#             "url": r[1],
#             "summary": r[2],
#             "score": float(r[3]) if r[3] is not None else None,
#         })

#     # --- BM25リランキング ---
#     query_tokens = tokenize(req.query)
#     docs_tokens = [tokenize(r[2]) for r in rows_summ]  # r[2] = summary text
#     bm25 = BM25Okapi(docs_tokens)
#     bm25_scores = bm25.get_scores(query_tokens)

#     rerank_summaries_out: List[Dict[str, Any]] = []
#     for r, score in sorted(zip(rows_summ, bm25_scores), key=lambda x: x[1], reverse=True):
#         rerank_summaries_out.append({
#             "doc_id": r[0],
#             "url": r[1],
#             "summary": r[2],
#             "score": float(score),
#         })

#     return {
#         "query": req.query,
#         "chunks": chunks_out,
#         "summaries": summaries_out,
#         "rerank_summaries": rerank_summaries_out,
#     }
