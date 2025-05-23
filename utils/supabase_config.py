"""
Supabase configuration and connection management.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

class SupabaseConfig:
    """Manages Supabase connection and configuration"""
    
    _instance: Optional['SupabaseConfig'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(SupabaseConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Supabase client"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL and key must be set as environment variables. "
                "Create a .env file with SUPABASE_URL and SUPABASE_KEY."
            )
        
        self._client = create_client(supabase_url, supabase_key)
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance"""
        if self._client is None:
            self._initialize()
        return self._client
    
    def get_table(self, table_name: str):
        """Get a reference to a Supabase table"""
        return self.client.table(table_name)
