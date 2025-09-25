CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks table (company/year metadata + text + embedding)
CREATE TABLE IF NOT EXISTS chunks (
  id              BIGSERIAL PRIMARY KEY,
  company         TEXT NOT NULL,
  year            INT  NOT NULL,
  chunk_id        INT  NOT NULL,
  source_doc      TEXT NOT NULL,
  text            TEXT NOT NULL,
  embedding       VECTOR(384)  -- 384 dims for all-MiniLM-L6-v2
);

-- Composite index for filtering
CREATE INDEX IF NOT EXISTS idx_chunks_meta ON chunks(company, year);

-- IVF index for fast ANN (requires analyze; build once data exists)
-- You can build this after loading:
-- CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
