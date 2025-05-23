# RAG System with Semantic Cache

A Retrieval-Augmented Generation (RAG) system with semantic caching, agentic routing, and Supabase integration.

## Features

- **Semantic Caching**: Cache similar queries to improve response time and reduce API costs
- **Agentic Routing**: Intelligently route queries to the appropriate search mechanism
- **Vector Search**: Use pgvector in Supabase for efficient similarity search
- **Web Search Integration**: Fallback to web search for recent information
- **Document Upload**: Upload documents to the knowledge base
- **Streamlit UI**: User-friendly interface for interacting with the system

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
   - Use the "Add Text" tab to add text content directly
   - View uploaded documents in the "View Documents" tab

3. **View cache statistics**:
   - See cache entries, hits, and hit rate in the sidebar

## License

MIT

## Acknowledgements

- [Supabase](https://supabase.com) for the vector database
- [OpenAI](https://openai.com) for the GPT-4.1 Mini model
- [Nomic AI](https://nomic.ai) for the embedding model
- [Brave Search](https://brave.com/search) for the web search API
- [Streamlit](https://streamlit.io) for the UI framework
