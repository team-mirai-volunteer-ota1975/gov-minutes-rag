-- Schema for crawled pages (run separately from the crawler)

CREATE TABLE IF NOT EXISTS crawled_pages (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    referrer_anchor_text TEXT,
    status_code INTEGER,
    content_type TEXT,
    content BYTEA,
    html_title TEXT,
    depth INTEGER,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique URL to avoid duplicates
CREATE UNIQUE INDEX IF NOT EXISTS crawled_pages_url_idx ON crawled_pages (url);

