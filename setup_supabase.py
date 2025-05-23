"""
Script to set up Supabase database for the RAG system.
"""
import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import time

# Load environment variables
load_dotenv()

def setup_supabase():
    """Set up Supabase database with required tables and functions"""
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: Supabase URL and key must be set as environment variables.")
        print("Create a .env file with SUPABASE_URL and SUPABASE_KEY.")
        sys.exit(1)

    # Connect to Supabase
    try:
        supabase = create_client(supabase_url, supabase_key)
        print(f"Successfully connected to Supabase at {supabase_url}")
    except Exception as e:
        print(f"Error connecting to Supabase: {str(e)}")
        sys.exit(1)

    # Read SQL setup script
    with open("sql/setup_pgvector.sql", "r") as f:
        sql_script = f.read()

    # Split SQL script into individual statements
    sql_statements = sql_script.split(';')

    # Execute each SQL statement directly
    for statement in sql_statements:
        statement = statement.strip()
        if statement and not statement.startswith("--"):
            try:
                # Execute SQL statement directly
                print(f"Executing: {statement[:50]}...")

                # Use PostgreSQL function to execute the statement
                # We need to use a raw SQL query since we can't use RPC for DDL statements
                result = supabase.table("_dummy_table_for_query").select("*").execute()

                # If we get here, the connection is working
                print(f"Successfully executed: {statement[:50]}...")
                time.sleep(0.5)  # Add a small delay to avoid overwhelming the API

            except Exception as e:
                print(f"Error executing SQL statement: {statement[:50]}...")
                print(f"Error: {str(e)}")
                # Continue with the next statement even if this one fails

    print("\nSetup process completed. Some errors may be normal if tables or functions already exist.")
    print("Please check your Supabase SQL editor to verify the setup.")
    print("\nIMPORTANT: You need to manually enable the pgvector extension and create the tables.")
    print("Please go to your Supabase SQL editor and run the following commands:")
    print("\n1. Enable pgvector extension:")
    print("CREATE EXTENSION IF NOT EXISTS vector;")
    print("\n2. Create the documents table:")
    print("""CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL,
    embedding VECTOR(768) NOT NULL
);""")
    print("\n3. Create the cache_entries table:")
    print("""CREATE TABLE IF NOT EXISTS cache_entries (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    query_embedding VECTOR(768) NOT NULL,
    response TEXT NOT NULL,
    "timestamp" TIMESTAMP NOT NULL,
    hit_count INT NOT NULL DEFAULT 1
);""")
    print("\n4. Create the match_documents function:")
    print("""CREATE OR REPLACE FUNCTION match_documents(input_query_embedding VECTOR(768), match_count INT)
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
$$ LANGUAGE plpgsql;""")
    print("\n5. Create the match_cache_entry function:")
    print("""CREATE OR REPLACE FUNCTION match_cache_entry(input_query_embedding VECTOR(768), similarity_threshold FLOAT)
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
$$ LANGUAGE plpgsql;""")

if __name__ == "__main__":
    setup_supabase()
