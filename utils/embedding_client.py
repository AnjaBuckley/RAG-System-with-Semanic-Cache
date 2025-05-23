"""
Embedding client for different embedding models.
"""
import os
from typing import List, Union, Optional
import numpy as np
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer

from utils.logging_utils import logger

# Load environment variables
load_dotenv()

class EmbeddingClient:
    """Base class for embedding clients"""
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        raise NotImplementedError("Subclasses must implement encode method")
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension"""
        raise NotImplementedError("Subclasses must implement embedding_dim property")


class SentenceTransformerClient(EmbeddingClient):
    """Client for sentence-transformers models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with model name"""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts)
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class NomicAIClient(EmbeddingClient):
    """Client for Nomic AI Atlas embedding model"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "nomic-embed-text-v1.5"):
        """Initialize with API key and model name"""
        self.api_key = api_key or os.getenv("NOMIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Nomic AI API key is required. Set NOMIC_API_KEY in .env file or pass as parameter."
            )
        
        self.model_name = model_name
        self.api_url = "https://api-atlas.nomic.ai/v1/embedding/text"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Nomic AI Atlas model dimension is 768
        self._embedding_dim = 768
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) to embeddings using Nomic AI API"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            payload = {
                "model": self.model_name,
                "texts": texts
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = np.array(result["embeddings"])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error calling Nomic AI API: {str(e)}")
            # Fallback to a default embedding (zeros)
            logger.warning(f"Falling back to zero embeddings")
            return np.zeros((len(texts), self.embedding_dim))
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension"""
        return self._embedding_dim


def get_embedding_client(client_type: str = "sentence_transformer") -> EmbeddingClient:
    """
    Factory function to get the appropriate embedding client
    
    Args:
        client_type: Type of client to use ('sentence_transformer' or 'nomic_ai')
        
    Returns:
        An embedding client instance
    """
    if client_type == "sentence_transformer":
        return SentenceTransformerClient()
    elif client_type == "nomic_ai":
        return NomicAIClient()
    else:
        raise ValueError(f"Unknown client type: {client_type}")
