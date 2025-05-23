"""
Supabase pgvector-based vector store for document retrieval.
"""
import json
from typing import List, Tuple, Dict, Any
import numpy as np

from models.data_models import Document
from utils.logging_utils import logger
from utils.supabase_config import SupabaseConfig
from utils.embedding_client import get_embedding_client, EmbeddingClient

class VectorStore:
    """Supabase pgvector-based vector store for document retrieval"""
    
    def __init__(self, embedding_dim: int = 768, table_name: str = "documents", embedding_client_type: str = "nomic_ai"):
        self.table_name = table_name
        self.encoder = get_embedding_client(embedding_client_type)
        self.embedding_dim = self.encoder.embedding_dim
        self.supabase = SupabaseConfig()
        
        # Ensure the table exists
        self._initialize_table()
    
    def _initialize_table(self):
        """Initialize the documents table in Supabase if it doesn't exist"""
        # Note: This assumes you've already created the table in Supabase
        # with the appropriate pgvector extension and schema
        # You would typically do this through Supabase UI or SQL migrations
        logger.info(f"Using Supabase table: {self.table_name}")
    
    def _document_to_record(self, doc: Document) -> Dict[str, Any]:
        """Convert a Document object to a Supabase record"""
        if doc.embedding is None:
            # Use the encoder to generate embeddings
            embedding = self.encoder.encode(doc.content)
            if len(embedding.shape) > 1:
                # If the encoder returns a batch (even for a single input), take the first one
                doc.embedding = embedding[0]
            else:
                doc.embedding = embedding
        
        # Convert numpy array to list for JSON serialization
        embedding_list = doc.embedding.tolist()
        
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": json.dumps(doc.metadata),
            "embedding": embedding_list
        }
    
    def _record_to_document(self, record: Dict[str, Any]) -> Document:
        """Convert a Supabase record to a Document object"""
        return Document(
            id=record["id"],
            content=record["content"],
            metadata=json.loads(record["metadata"]) if isinstance(record["metadata"], str) else record["metadata"],
            embedding=np.array(record["embedding"]) if record.get("embedding") else None
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        records = [self._document_to_record(doc) for doc in documents]
        
        # Insert records into Supabase
        # Using upsert to handle duplicates based on id
        result = self.supabase.get_table(self.table_name).upsert(records).execute()
        
        logger.info(f"Added {len(documents)} documents to Supabase vector store")
        return result
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for relevant documents using pgvector similarity search"""
        # Generate query embedding
        embedding = self.encoder.encode(query)
        if len(embedding.shape) > 1:
            # If the encoder returns a batch (even for a single input), take the first one
            query_embedding = embedding[0].tolist()
        else:
            query_embedding = embedding.tolist()
        
        # Use pgvector's cosine similarity search
        # The SQL function would be something like:
        # SELECT *, embedding <=> $1 as distance FROM documents ORDER BY distance LIMIT $2
        result = self.supabase.client.rpc(
            'match_documents',
            {
                'input_query_embedding': query_embedding,
                'match_count': top_k
            }
        ).execute()
        
        if hasattr(result, 'data') and result.data:
            results = []
            for item in result.data:
                doc = self._record_to_document(item)
                similarity_score = 1.0 - item.get('distance', 0)  # Convert distance to similarity
                results.append((doc, similarity_score))
            return results
        
        return []
        
    def get_all_documents(self, limit: int = 100) -> List[Document]:
        """Get all documents from the vector store
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of Document objects
        """
        try:
            # Query Supabase for all documents up to the limit
            result = self.supabase.get_table(self.table_name).select("*").limit(limit).execute()
            
            if hasattr(result, 'data') and result.data:
                documents = [self._record_to_document(record) for record in result.data]
                logger.info(f"Retrieved {len(documents)} documents from vector store")
                return documents
            
            logger.info("No documents found in vector store")
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
