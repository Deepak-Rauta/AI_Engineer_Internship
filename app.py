import streamlit as st
import logging
from typing import Dict, Any
import traceback

# Import our modules
from config.config import Config
from models.llm import get_gemini_llm
from utils.rag_pipeline import get_rag_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stats-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ChatbotApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        """Initialize the chatbot application"""
        try:
            # Validate configuration
            Config.validate_config()
            
            # Initialize components
            self.llm = get_gemini_llm()
            self.rag_pipeline = get_rag_pipeline()
            
            # Initialize session state
            self._initialize_session_state()
            
            logger.info("Chatbot application initialized successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            st.stop()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'response_mode' not in st.session_state:
            st.session_state.response_mode = "detailed"
        
        if 'use_web_search' not in st.session_state:
            st.session_state.use_web_search = True
        
        if 'knowledge_base_stats' not in st.session_state:
            st.session_state.knowledge_base_stats = {}
    
    def run(self):
        """Run the main application"""
        try:
            # Header
            st.markdown("""
            <div class="main-header">
                <h1>🤖 RAG Chatbot - AI Assistant</h1>
                <p>Intelligent conversations powered by Gemini AI, RAG, and Web Search</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sidebar
            self._render_sidebar()
            
            # Main chat interface
            self._render_chat_interface()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
    
    def _render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Response mode selection
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 💬 Response Mode")
            response_mode = st.radio(
                "Choose response style:",
                ["detailed", "concise"],
                index=0 if st.session_state.response_mode == "detailed" else 1,
                help="Detailed: Comprehensive responses with explanations\nConcise: Brief, summarized answers"
            )
            st.session_state.response_mode = response_mode
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Web search toggle
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("### 🌐 Web Search")
            use_web_search = st.checkbox(
                "Enable web search fallback",
                value=st.session_state.use_web_search,
                help="Use web search when document knowledge is insufficient"
            )
            st.session_state.use_web_search = use_web_search
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Document upload section
            self._render_document_upload()
            
            # Knowledge base management
            self._render_knowledge_base_management()
            
            # Statistics
            self._render_statistics()
    
    def _render_document_upload(self):
        """Render document upload section"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="Upload documents to enhance the chatbot's knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        results = self.rag_pipeline.add_documents_from_files(uploaded_files)
                        
                        # Display results
                        if results['total_chunks'] > 0:
                            st.success(f"✅ Processed {results['total_chunks']} chunks from {len(results['processed_files'])} files")
                            
                            # Show file details
                            for file_info in results['processed_files']:
                                if file_info['status'] == 'success':
                                    st.info(f"📄 {file_info['name']}: {file_info['chunks']} chunks")
                                else:
                                    st.error(f"❌ {file_info['name']}: {file_info.get('error', 'Unknown error')}")
                        
                        if results['errors']:
                            for error in results['errors']:
                                st.error(f"❌ {error}")
                        
                        # Update stats
                        st.session_state.knowledge_base_stats = self.rag_pipeline.get_pipeline_stats()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_knowledge_base_management(self):
        """Render knowledge base management section"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🗄️ Knowledge Base")
        
        # Get current stats
        stats = self.rag_pipeline.get_pipeline_stats()
        num_docs = stats.get('vector_store', {}).get('num_documents', 0)
        
        st.info(f"📊 Documents in knowledge base: {num_docs}")
        
        if num_docs > 0:
            if st.button("🗑️ Clear Knowledge Base", help="Remove all documents from the knowledge base"):
                try:
                    self.rag_pipeline.clear_knowledge_base()
                    st.success("Knowledge base cleared successfully!")
                    st.session_state.knowledge_base_stats = self.rag_pipeline.get_pipeline_stats()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing knowledge base: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_statistics(self):
        """Render application statistics"""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Statistics")
        
        try:
            stats = self.rag_pipeline.get_pipeline_stats()
            
            # Vector store stats
            vector_stats = stats.get('vector_store', {})
            st.metric("Documents", vector_stats.get('num_documents', 0))
            st.metric("Embedding Dimension", vector_stats.get('embedding_dimension', 0))
            
            # Configuration status
            web_search_status = "✅ Enabled" if stats.get('web_search_configured', False) else "❌ Not Configured"
            st.info(f"🌐 Web Search: {web_search_status}")
            
            embedding_model = stats.get('embedding_model', 'Unknown')
            st.info(f"🧠 Embedding Model: {embedding_model}")
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_chat_interface(self):
        """Render the main chat interface"""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        st.text(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = self._generate_response(prompt)
                        st.markdown(response)
                        
                        # Show sources if available
                        if sources and sources.strip() and "No sources found" not in sources:
                            with st.expander("📚 Sources"):
                                st.text(sources)
                        
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                        logger.error(f"Response generation error: {str(e)}")
    
    def _generate_response(self, query: str) -> tuple[str, str]:
        """
        Generate response using RAG pipeline and LLM
        
        Args:
            query: User query
            
        Returns:
            Tuple of (response, sources)
        """
        try:
            logger.info(f"[CHAT DEBUG] Processing query: {query}")
            
            # Retrieve context using RAG pipeline
            context, sources = self.rag_pipeline.retrieve_context(
                query, 
                use_web_search=st.session_state.use_web_search
            )
            
            logger.info(f"[CHAT DEBUG] Context length: {len(context)} characters")
            logger.info(f"[CHAT DEBUG] Sources: {sources}")
            
            if context and len(context) > 50:
                st.info(f"🔍 Found relevant context from your documents ({len(context)} characters)")
            elif not context:
                st.warning("⚠️ No relevant context found in uploaded documents. Using general knowledge.")
            
            # Generate response using LLM
            response = self.llm.generate_response(
                prompt=query,
                context=context,
                response_mode=st.session_state.response_mode
            )
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def main():
    """Main application entry point"""
    try:
        app = ChatbotApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start application: {str(e)}")
        logger.error(f"Application startup error: {str(e)}")

if __name__ == "__main__":
    main()
