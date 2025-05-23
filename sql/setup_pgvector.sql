-- Enable the pgvector extension
-- Note: This is now handled directly in the setup_supabase.py script
-- CREATE EXTENSION IF NOT EXISTS vector;

-- Create a function to execute SQL statements from the setup script
-- Note: This is now handled directly in the setup_supabase.py script
-- CREATE OR REPLACE FUNCTION exec_sql(sql TEXT)
-- RETURNS TEXT AS $$
-- BEGIN
--     EXECUTE sql;
--     RETURN 'SQL executed successfully';
-- EXCEPTION WHEN OTHERS THEN
--     RETURN 'Error: ' || SQLERRM;
-- END;
-- $$ LANGUAGE plpgsql;

-- Create documents table with vector support
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(768) NOT NULL
);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_documents(input_query_embedding VECTOR(768), match_count INT)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(768),
    distance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        d.embedding,
        1 - (d.embedding <=> input_query_embedding) AS distance
    FROM
        documents d
    ORDER BY
        d.embedding <=> input_query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Create cache_entries table
CREATE TABLE IF NOT EXISTS cache_entries (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    query_embedding VECTOR(768) NOT NULL,
    response TEXT NOT NULL,
    "timestamp" TIMESTAMP NOT NULL,
    hit_count INT NOT NULL DEFAULT 1
);

-- Create function for semantic cache lookup
CREATE OR REPLACE FUNCTION match_cache_entry(input_query_embedding VECTOR(768), similarity_threshold FLOAT)
RETURNS TABLE (
    query_hash TEXT,
    query TEXT,
    response TEXT,
    "timestamp" TIMESTAMP,
    hit_count INT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.query_hash,
        c.query,
        c.response,
        c."timestamp",
        c.hit_count,
        1 - (c.query_embedding <=> input_query_embedding) AS similarity
    FROM
        cache_entries c
    WHERE
        1 - (c.query_embedding <=> input_query_embedding) >= similarity_threshold
    ORDER BY
        similarity DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;
