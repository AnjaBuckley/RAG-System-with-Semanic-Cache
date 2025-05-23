# RAG System with Semantic Cache

A Retrieval-Augmented Generation (RAG) system with semantic caching, agentic routing, and Supabase integration.

## Features

- **Semantic Caching**: Cache similar queries to improve response time and reduce API costs
  - Smart cache invalidation for time-sensitive queries (e.g., different years)
  - Adjustable similarity threshold via the UI
  - Cache statistics and manual clearing option
- **Agentic Routing**: Intelligently route queries to the appropriate search mechanism
  - Year-aware routing for current/future year queries
  - Fallback to web search when local documents lack relevant information
- **Vector Search**: Use pgvector in Supabase for efficient similarity search
- **Web Search Integration**: Brave Search API integration for recent information
- **Document Upload**: Upload documents (TXT, PDF, DOCX) to the knowledge base
  - PDF processing with automatic page splitting for large documents
  - Page-level metadata and content extraction
- **Streamlit UI**: User-friendly interface for interacting with the system
- **Text Cleaning**: Automatic formatting correction for web search results

## Architecture

The system consists of the following components:

- **RAG Pipeline**: Main orchestrator that coordinates all components
- **Vector Store**: Supabase pgvector-based document storage and retrieval
- **Semantic Cache**: Caches similar queries and responses
- **Agentic Router**: Routes queries to the appropriate search mechanism
- **Web Searcher**: Integrates with Brave Search API for web search
- **Embedding Client**: Generates embeddings for documents and queries

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-system-with-semantic-cache.git
   cd rag-system-with-semantic-cache
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Supabase**:
   - Create a Supabase account and project at [supabase.com](https://supabase.com)
   - Enable the pgvector extension in your Supabase project
   - Create a `.env` file with your Supabase credentials:
     ```
     SUPABASE_URL=your_supabase_url
     SUPABASE_KEY=your_supabase_key
     ```
   - Run the setup script:
     ```bash
     python setup_supabase.py
     ```
   - Follow the instructions to manually create the necessary tables and functions in the Supabase SQL editor

5. **Set up API keys**:
   - Add the following to your `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     BRAVE_API_KEY=your_brave_api_key
     NOMIC_API_KEY=your_nomic_api_key
     ```
   - Note: The system uses OpenAI's GPT-4o model with the Chat Completions API. If you don't have access to this model, the system will automatically fall back to GPT-3.5-turbo or provide a generated response based on retrieved documents.

6. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Search for information**:
   - Enter a query in the search box
   - Toggle web search on/off as needed
   - View the answer, sources, and metadata

2. **Upload documents**:
   - Use the "Upload File" tab to upload text, PDF, or Word documents
     - PDF files are automatically processed and large PDFs (>5 pages) are split into individual pages
     - Each page is stored separately with page metadata for better retrieval
   - Use the "Add Text" tab to add text content directly
   - View uploaded documents in the "View Documents" tab
     - PDF pages are displayed with page numbers and file information

3. **Manage cache**:
   - See cache entries, hits, and hit rate in the sidebar
   - Adjust the similarity threshold to control cache strictness
   - Clear the cache manually when needed
   - View detailed cache entries to understand what's being cached

## License

MIT

## Acknowledgements

- [Supabase](https://supabase.com) for the vector database
- [OpenAI](https://openai.com) for the GPT-4o model
- [Nomic AI](https://nomic.ai) for the embedding model
- [Brave Search](https://brave.com/search) for the web search API
- [Streamlit](https://streamlit.io) for the UI framework
