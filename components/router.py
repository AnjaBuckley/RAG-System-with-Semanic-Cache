"""
Agenting router for query routing.
"""
from utils.logging_utils import logger

class AgenticRouter:
    """Routes queries to appropriate search mechanism"""
    
    def __init__(self):
        self.local_keywords = {
            'company', 'revenue', 'profit', 'earnings', 'financial', 'balance sheet',
            'income statement', 'cash flow', 'assets', 'liabilities', 'equity',
            'sec filing', '10-k', 'annual report', 'quarterly', 'fiscal year'
        }
        
        self.web_keywords = {
            'latest', 'recent', 'current', 'today', 'news', 'market cap',
            'stock price', 'breaking', 'announcement', 'update'
        }
    
    def route_query(self, query: str) -> str:
        """Determine routing strategy for query"""
        query_lower = query.lower()
        
        # Score for local vs web search
        local_score = sum(1 for keyword in self.local_keywords if keyword in query_lower)
        web_score = sum(1 for keyword in self.web_keywords if keyword in query_lower)
        
        # Check for explicit web indicators
        if any(term in query_lower for term in ['latest', 'current', 'today', 'recent']):
            web_score += 2
        
        # Route decision
        if web_score > local_score:
            logger.info(f"ROUTING: Web search (web_score: {web_score}, local_score: {local_score})")
            return "web_search"
        else:
            logger.info(f"ROUTING: Local search (web_score: {web_score}, local_score: {local_score})")
            return "local_search"
