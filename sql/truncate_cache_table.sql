-- Function to truncate the cache table
-- This function is safer than allowing direct DELETE operations
-- as it can be controlled with proper permissions

-- Create the function
CREATE OR REPLACE FUNCTION truncate_cache_table(table_name text)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER -- Run with the privileges of the function creator
AS $$
BEGIN
  -- Validate the table name to prevent SQL injection
  -- Only allow specific tables to be truncated
  IF table_name = 'cache_entries' THEN
    -- Use EXECUTE for dynamic SQL
    EXECUTE 'TRUNCATE TABLE ' || quote_ident(table_name);
  ELSE
    RAISE EXCEPTION 'Unauthorized table: %', table_name;
  END IF;
END;
$$;

-- Grant execute permission to the function
-- Replace 'authenticated' with the appropriate role if needed
GRANT EXECUTE ON FUNCTION truncate_cache_table(text) TO authenticated;

-- Comment explaining the function
COMMENT ON FUNCTION truncate_cache_table(text) IS 'Safely truncates the specified cache table. Only works with authorized tables.';
