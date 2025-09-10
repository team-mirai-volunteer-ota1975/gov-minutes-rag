-- RAG schema for government meeting minutes
-- Requires: PostgreSQL 14+ and pgvector 0.7+

-- Base table (reference, managed by crawler)
-- Provided for reference; not created by this project.
-- CREATE TABLE IF NOT EXISTS crawled_pages (
--     id BIGSERIAL PRIMARY KEY,
--     url TEXT NOT NULL,
--     referrer_anchor_text TEXT,
--     status_code INTEGER,
--     content_type TEXT,
--     content BYTEA,
--     html_title TEXT,
--     depth INTEGER,
--     fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
-- );

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Meeting metadata (normalized)
CREATE TABLE IF NOT EXISTS meeting_metadata (
    doc_id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    ministry TEXT,
    council_name TEXT,
    meeting_no TEXT,
    date DATE,
    location TEXT,
    attendees JSONB,
    agenda JSONB,
    discussion_text TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_meeting_metadata_url ON meeting_metadata(url);

-- Meeting chunks (for RAG search)
CREATE TABLE IF NOT EXISTS meeting_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    doc_id UUID REFERENCES meeting_metadata(doc_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS idx_meeting_chunks_doc_id ON meeting_chunks(doc_id);

-- HNSW index for vector search (pgvector >= 0.7)
CREATE INDEX IF NOT EXISTS idx_meeting_chunks_embedding_hnsw
ON meeting_chunks USING hnsw (embedding vector_cosine_ops);

-- Optional: trigram index for hybrid keyword search
CREATE INDEX IF NOT EXISTS idx_meeting_chunks_trgm ON meeting_chunks USING gin (chunk_text gin_trgm_ops);

