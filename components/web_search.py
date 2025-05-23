"""
Brave Search API integration for web search functionality.
"""
import os
import json
import requests
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from utils.logging_utils import logger

# Load environment variables
load_dotenv()

class BraveWebSearcher:
    """Web search using Brave Search API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key from env or parameter"""
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Brave Search API key is required. Set BRAVE_API_KEY in .env file or pass as parameter."
            )
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }

    def search(self, query: str, count: int = 5) -> str:
        """
        Search the web using Brave Search API

        Args:
            query: The search query
            count: Number of results to return (max 20)

        Returns:
            Formatted search results as a string
        """
        params = {
            "q": query,
            "count": min(count, 20),  # Brave API max is 20
            "search_lang": "en"
        }

        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()

            data = response.json()
            return self._format_results(data, query)

        except requests.RequestException as e:
            logger.error(f"Brave Search API error: {str(e)}")
            # Fallback to mock results if API fails
            return self._mock_fallback_search(query)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing excessive whitespace and fixing common formatting issues"""
        import re

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Fix broken numbers (e.g., "350 b i l l i o n" -> "350 billion")
        text = re.sub(r'(\d+)\s+([b|m|t])\s+i\s+l\s+l\s+i\s+o\s+n', r'\1 \2illion', text)
        text = re.sub(r'(\d+)\s+billion', r'\1 billion', text)
        text = re.sub(r'(\d+)\s+million', r'\1 million', text)
        text = re.sub(r'(\d+)\s+trillion', r'\1 trillion', text)

        # Fix broken percentages
        text = re.sub(r'(\d+)\s*\.\s*(\d+)\s*%', r'\1.\2%', text)

        # Remove any remaining unusual character patterns
        text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', text)  # Add space after period between words

        return text.strip()

    def _format_results(self, data: Dict[str, Any], query: str) -> str:
        """Format the API response into a readable string"""
        if not data.get("web") or not data["web"].get("results"):
            return f"Web Search Results:\nNo results found for query: '{query}'"

        results = data["web"]["results"]
        formatted_results = []

        for result in results:
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            description = result.get('description', 'No description')

            # Clean the description text
            description = self._clean_text(description)

            formatted_results.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}"
            )

        return "Web Search Results:\n\n" + "\n\n".join(formatted_results)

    def _mock_fallback_search(self, query: str) -> str:
        """Fallback to mock results if the API fails"""
        logger.warning("Using mock fallback search due to API failure")

        mock_results = {
            "nvidia": "NVIDIA Corporation reported record quarterly revenue of $60.9 billion for Q3 2024, up 206% year-over-year, driven by Data Center revenue of $51.0 billion.",
            "tesla": "Tesla's Q3 2024 earnings showed revenue of $25.2 billion, with vehicle deliveries reaching 462,890 units.",
            "apple": "Apple reported Q4 2024 revenue of $94.9 billion, with iPhone revenue of $46.2 billion.",
            "microsoft": "Microsoft's Q1 2024 revenue reached $56.5 billion, with Azure and cloud services growing 33%."
        }

        query_lower = query.lower()
        for company, result in mock_results.items():
            if company in query_lower:
                return f"Web Search Results (MOCK FALLBACK):\n{result}\n\nSource: Mock financial data service"

        return f"Web Search Results (MOCK FALLBACK):\nNo specific recent data found for query: '{query}'"


# For backward compatibility
MockWebSearcher = BraveWebSearcher
