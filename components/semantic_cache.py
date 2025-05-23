"""
Semantic caching system using Supabase for persistence.
"""

import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from models.data_models import CacheEntry
from utils.logging_utils import logger
from utils.supabase_config import SupabaseConfig
from utils.embedding_client import get_embedding_client, EmbeddingClient


class SemanticCache:
    """Semantic caching system using Supabase for persistence"""

    def __init__(
        self,
        similarity_threshold: float = 0.98,
        table_name: str = "cache_entries",
        embedding_client_type: str = "nomic_ai",
    ):
        self.similarity_threshold = similarity_threshold
        self.table_name = table_name
        self.encoder = get_embedding_client(embedding_client_type)
        self.supabase = SupabaseConfig()

        # Local cache for faster access (optional)
        self.local_cache: Dict[str, CacheEntry] = {}

        # Ensure the table exists
        self._initialize_table()

    def _initialize_table(self):
        """Initialize the cache table in Supabase if it doesn't exist"""
        # Note: This assumes you've already created the table in Supabase
        # with the appropriate pgvector extension and schema
        # You would typically do this through Supabase UI or SQL migrations
        logger.info(f"Using Supabase table: {self.table_name}")

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding"""
        embedding = self.encoder.encode(query)
        if len(embedding.shape) > 1:
            # If the encoder returns a batch (even for a single input), take the first one
            return embedding[0]
        return embedding

    def _record_to_cache_entry(self, record: Dict[str, Any]) -> CacheEntry:
        """Convert a Supabase record to a CacheEntry object"""
        return CacheEntry(
            query=record["query"],
            query_embedding=np.array(record["query_embedding"])
            if record.get("query_embedding")
            else None,
            response=record["response"],
            timestamp=datetime.fromisoformat(record["timestamp"])
            if isinstance(record["timestamp"], str)
            else record["timestamp"],
            hit_count=record["hit_count"],
        )

    def get(self, query: str) -> Optional[Tuple[str, bool]]:
        """Retrieve from cache if similar query exists"""
        query_embedding = self._encode_query(query).tolist()

        # Extract years from the query
        import re

        years_in_query = set(re.findall(r"\b(19\d\d|20\d\d)\b", query))

        # Use pgvector's similarity search to find the best match
        result = self.supabase.client.rpc(
            "match_cache_entry",
            {
                "input_query_embedding": query_embedding,
                "similarity_threshold": self.similarity_threshold,
            },
        ).execute()

        if hasattr(result, "data") and result.data:
            # We found a match
            match = result.data[0]
            query_hash = match["query_hash"]
            similarity = match["similarity"]
            cached_query = match["query"]

            # Extract years from the cached query
            years_in_cached = set(re.findall(r"\b(19\d\d|20\d\d)\b", cached_query))

            # Log detailed information
            logger.info(f"Cache match found with similarity {similarity:.3f}")
            logger.info(f"Current query: '{query}'")
            logger.info(f"Cached query: '{cached_query}'")
            logger.info(f"Years in current query: {years_in_query}")
            logger.info(f"Years in cached query: {years_in_cached}")

            # Check if the queries mention different years
            if years_in_query and years_in_cached and years_in_query != years_in_cached:
                logger.info(
                    f"Cache MISS: Different years mentioned ({years_in_query} vs {years_in_cached})"
                )
                return None, False

            # Check if one query mentions a year and the other doesn't
            if (years_in_query and not years_in_cached) or (
                not years_in_query and years_in_cached
            ):
                logger.info(
                    f"Cache MISS: Year mismatch (one query mentions a year, the other doesn't)"
                )
                return None, False

            # Update hit count
            self.supabase.get_table(self.table_name).update(
                {"hit_count": match["hit_count"] + 1}
            ).eq("query_hash", query_hash).execute()

            logger.info(f"Cache HIT: {similarity:.3f} similarity")
            return match["response"], True

        logger.info("Cache MISS: No matching entries found")
        return None, False

    def put(self, query: str, response: str):
        """Store in cache"""
        query_hash = self._get_query_hash(query)
        query_embedding = self._encode_query(query).tolist()

        # Insert into Supabase
        record = {
            "query_hash": query_hash,
            "query": query,
            "query_embedding": query_embedding,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "hit_count": 1,
        }

        self.supabase.get_table(self.table_name).upsert(record).execute()
        logger.info(f"Cached response for query: {query[:50]}...")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        # Get all cache entries
        result = self.supabase.get_table(self.table_name).select("*").execute()

        if not hasattr(result, "data") or not result.data:
            return {"total_entries": 0, "total_hits": 0, "hit_rate": 0}

        entries = result.data
        total_entries = len(entries)
        total_hits = sum(
            entry["hit_count"] - 1 for entry in entries
        )  # -1 because first is not a hit
        total_queries = total_entries + total_hits

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "hit_rate": total_hits / total_queries if total_queries > 0 else 0,
        }

    def clear_cache(self) -> bool:
        """Clear all entries from the cache

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all entries to count them
            result = self.supabase.get_table(self.table_name).select("*").execute()

            if hasattr(result, "data") and result.data:
                entries = result.data
                logger.info(f"Found {len(entries)} entries to delete")

                # Simple approach: delete all entries
                try:
                    # Try a simple delete without any conditions
                    # This should work with basic permissions
                    self.supabase.get_table(self.table_name).delete().execute()
                    logger.info("Cache cleared successfully using delete")
                    return True
                except Exception as delete_error:
                    logger.warning(
                        f"Delete all failed: {str(delete_error)}. Trying one by one."
                    )

                    # If bulk delete fails, try deleting one by one
                    success_count = 0
                    for entry in entries:
                        try:
                            if "query_hash" in entry:
                                self.supabase.get_table(self.table_name).delete().eq(
                                    "query_hash", entry["query_hash"]
                                ).execute()
                                success_count += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete entry {entry.get('query_hash')}: {str(e)}"
                            )

                    logger.info(
                        f"Deleted {success_count} out of {len(entries)} entries"
                    )
                    return success_count > 0
            else:
                logger.info("No entries found to delete")
                return True

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
