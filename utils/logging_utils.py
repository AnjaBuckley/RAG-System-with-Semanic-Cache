"""
Logging utilities for the RAG system.
"""
import logging

def setup_logger():
    """Configure logging"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# Create a logger instance
logger = setup_logger()
