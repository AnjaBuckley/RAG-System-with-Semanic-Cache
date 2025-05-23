"""
Main Streamlit application for the RAG system.
"""
import streamlit as st
from pipeline.rag_pipeline import RAGPipeline

def main():
    st.set_page_config(
        page_title="Advanced RAG Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 Advanced RAG Search Engine")
    st.markdown("*Retrieval-Augmented Generation with Semantic Caching and Agentic Routing*")

    # Initialize the RAG pipeline and session state variables
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    # Initialize other session state variables if they don't exist
    if 'clear_text_content' not in st.session_state:
        st.session_state.clear_text_content = False

    if 'clear_uploaded_file' not in st.session_state:
        st.session_state.clear_uploaded_file = False

    if 'show_documents' not in st.session_state:
        st.session_state.show_documents = False

    # Sidebar for settings and stats
    with st.sidebar:
        st.header("⚙️ Settings")

        # Web search toggle
        allow_web_search = st.toggle("Enable Web Search", value=True)

        # Cache settings
        st.subheader("Cache Settings")
        current_threshold = st.session_state.rag_pipeline.cache.similarity_threshold
        new_threshold = st.slider("Similarity Threshold", min_value=0.80, max_value=1.0, value=current_threshold, step=0.01,
                                 help="Higher values make the cache more strict (fewer hits)")

        # Update the threshold if it changed
        if new_threshold != current_threshold:
            st.session_state.rag_pipeline.cache.similarity_threshold = new_threshold
            st.success(f"Similarity threshold updated to {new_threshold:.2f}")

        # Cache statistics
        st.header("📊 Cache Statistics")
        cache_stats = st.session_state.rag_pipeline.cache.get_stats()

        st.metric("Cache Entries", cache_stats["total_entries"])
        st.metric("Cache Hits", cache_stats["total_hits"])
        st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1%}")

        # Add a button to clear the cache
        if st.button("Clear Cache"):
            if st.session_state.rag_pipeline.cache.clear_cache():
                st.success("Cache cleared successfully!")
                st.rerun()  # Refresh the page to update the stats
            else:
                st.error("Failed to clear the cache.")

        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This RAG system features:
        - Semantic caching for faster responses
        - Agentic query routing
        - Vector search with pgvector
        - Web search integration
        - Document upload capabilities
        """)

    # Main search interface
    query = st.text_input("Enter your question:", placeholder="What was NVIDIA's revenue in 2023?")

    if query:
        with st.spinner("Searching..."):
            results = st.session_state.rag_pipeline.search(query, allow_web_search)

        # Display answer
        st.markdown("### Answer")
        st.markdown(results["answer"])

        # Display sources
        if results["sources"]:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(results["sources"]):
                    st.markdown(f"**Source {i+1}** (Score: {source['score']:.2f})")
                    st.markdown(f"```\n{source['content']}\n```")
                    st.json(source["metadata"])
                    st.divider()

        # Display web results if used
        if results.get("web_results"):
            with st.expander("🌐 Web Search Results", expanded=False):
                st.markdown(results["web_results"])

        # Display metadata
        with st.expander("🔍 Query Metadata", expanded=False):
            metadata = {
                "Cache Hit": results["cache_hit"],
                "Response Time": f"{results['response_time']:.2f} seconds",
                "Routing Decision": results["routing_decision"],
                "Web Search Used": results["web_search_used"]
            }
            st.json(metadata)

    # Document management section
    st.header("📄 Document Management")

    # Create tabs for different upload methods
    upload_tab, text_tab, view_tab = st.tabs(["Upload File", "Add Text", "View Documents"])

    with upload_tab:
        st.subheader("Upload a File")

        # Clear the file uploader if needed
        if 'clear_uploaded_file' in st.session_state and st.session_state.clear_uploaded_file:
            st.session_state.clear_uploaded_file = False
            # The file uploader will be reset on the next rerun

        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "doc", "docx"])

        # Optional metadata
        st.subheader("Document Metadata (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title", key="file_title")
        with col2:
            author = st.text_input("Author", key="file_author")

        # Upload button
        if uploaded_file is not None:
            if st.button("Upload Document", key="upload_file_btn"):
                try:
                    # Prepare metadata
                    metadata = {
                        "title": title if title else uploaded_file.name,
                        "author": author if author else "Unknown",
                        "source": "file_upload"
                    }

                    # Upload the file
                    doc_id = st.session_state.rag_pipeline.upload_file(
                        uploaded_file,
                        uploaded_file.name,
                        uploaded_file.type,
                        metadata
                    )

                    st.success(f"Document uploaded successfully! Document ID: {doc_id}")

                    # Reset the file uploader by setting a flag to clear it on next rerun
                    st.session_state.clear_uploaded_file = True
                    st.rerun()  # Force a rerun to clear the uploader

                except Exception as e:
                    st.error(f"Error uploading document: {str(e)}")

    with text_tab:
        st.subheader("Add Text Document")

        # Initialize text content value
        if 'clear_text_content' in st.session_state and st.session_state.clear_text_content:
            st.session_state.clear_text_content = False
            st.session_state.text_content = ""

        # Text input
        doc_content = st.text_area("Document Content", height=200, key="text_content")

        # Optional metadata
        st.subheader("Document Metadata (Optional)")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title", key="text_title")
        with col2:
            author = st.text_input("Author", key="text_author")

        # Add button
        if st.button("Add Document", key="add_text_btn"):
            if doc_content:
                try:
                    # Prepare metadata
                    metadata = {
                        "title": title if title else "Text Document",
                        "author": author if author else "Unknown",
                        "source": "text_input"
                    }

                    # Upload the text document
                    doc_id = st.session_state.rag_pipeline.upload_text_document(doc_content, metadata)

                    st.success(f"Document added successfully! Document ID: {doc_id}")

                    # Clear the text area by setting a flag to clear it on next rerun
                    st.session_state.clear_text_content = True
                    st.rerun()  # Force a rerun to clear the text area

                except Exception as e:
                    st.error(f"Error adding document: {str(e)}")
            else:
                st.warning("Please enter some content for the document.")

    with view_tab:
        st.subheader("View Uploaded Documents")

        if st.button("Refresh Documents"):
            st.session_state.show_documents = True

        if st.session_state.show_documents:
            with st.spinner("Loading documents..."):
                documents = st.session_state.rag_pipeline.get_all_documents()

            if documents:
                for i, doc in enumerate(documents):
                    with st.expander(f"Document: {doc['metadata'].get('title', doc['id'])}", expanded=False):
                        st.markdown(f"**ID:** {doc['id']}")
                        st.markdown(f"**Content Preview:**")
                        st.markdown(f"```\n{doc['content']}\n```")
                        st.markdown("**Metadata:**")
                        st.json(doc["metadata"])
            else:
                st.info("No documents found. Upload some documents first!")

if __name__ == "__main__":
    main()
