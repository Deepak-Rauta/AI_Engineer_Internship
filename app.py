"""
Main Streamlit application for RAG Chatbot
Professional UI with document upload and chat interface
"""
import streamlit as st
import os
from typing import List, Dict, Any
import logging
from datetime import datetime

# Import our modules
from config import config
from utils import doc_processor, rag_pipeline
from models import embedding_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import FontAwesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .chat-container {
        background: #f8fafc;
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .message-bubble {
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: #1f2937;
        color: white;
        margin-left: auto;
        margin-right: 0;
        text-align: right;
    }
    
    .bot-message {
        background: #8b5cf6;
        color: white;
        margin-left: 0;
        margin-right: auto;
    }
    
    .upload-section {
        background: #f1f5f9;
        border: 2px dashed #8b5cf6;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #1f2937;
        background: #e2e8f0;
    }
    
    .file-info {
        background: #e0e7ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #8b5cf6;
    }
    
    .response-mode-toggle {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stats-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .icon {
        margin-right: 0.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .message-bubble {
            max-width: 95%;
        }
        
        .main-header {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'response_mode' not in st.session_state:
        st.session_state.response_mode = "detailed"

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1><i class="fas fa-robot icon"></i>RAG Document Q&A Chatbot</h1>
        <p><i class="fas fa-info-circle icon"></i>Upload documents and ask questions with AI-powered search</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.markdown("### <i class='fas fa-cog icon'></i>Settings", unsafe_allow_html=True)
        
        # Response mode toggle
        st.markdown("#### <i class='fas fa-toggle-on icon'></i>Response Mode", unsafe_allow_html=True)
        response_mode = st.radio(
            "Choose response style:",
            ["detailed", "concise"],
            index=0 if st.session_state.response_mode == "detailed" else 1,
            help="Detailed: Comprehensive answers with explanations\nConcise: Brief, direct answers"
        )
        st.session_state.response_mode = response_mode
        
        # File upload section
        st.markdown("#### <i class='fas fa-upload icon'></i>Document Upload", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, MD"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Document statistics
        render_document_stats()
        
        # Clear chat button
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add to RAG system"""
    try:
        new_files = [f for f in uploaded_files if f.name not in [uf.name for uf in st.session_state.uploaded_files]]
        
        if new_files:
            with st.spinner("Processing documents..."):
                all_chunks = []
                
                for file in new_files:
                    try:
                        chunks = doc_processor.process_uploaded_file(file)
                        all_chunks.extend(chunks)
                        st.success(f"✅ Processed: {file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                
                if all_chunks:
                    # Add existing documents
                    if hasattr(embedding_model, 'documents') and embedding_model.documents:
                        all_chunks.extend(embedding_model.documents)
                    
                    # Build new index
                    rag_pipeline.add_documents(all_chunks)
                    st.session_state.documents_loaded = True
                    st.session_state.uploaded_files = uploaded_files
                    
                    st.success(f"🎉 Successfully processed {len(new_files)} files!")
                    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        logger.error(f"File processing error: {e}")

def render_document_stats():
    """Render document statistics in sidebar"""
    st.markdown("#### <i class='fas fa-chart-bar icon'></i>Document Stats", unsafe_allow_html=True)
    
    if st.session_state.documents_loaded and hasattr(embedding_model, 'documents'):
        num_docs = len(embedding_model.documents)
        st.metric("📄 Document Chunks", num_docs)
        
        if st.session_state.uploaded_files:
            st.metric("📁 Files Uploaded", len(st.session_state.uploaded_files))
            
            # Show file list
            st.markdown("*Uploaded Files:*")
            for file in st.session_state.uploaded_files:
                st.markdown(f"• {file.name}")
    else:
        st.info("No documents uploaded yet")

def render_chat_interface():
    """Render the main chat interface"""
    st.markdown("### <i class='fas fa-comments icon'></i>Chat Interface", unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            render_message(message)
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        handle_user_input(user_input)

def render_message(message: Dict[str, Any]):
    """Render a single chat message"""
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-bubble user-message">
            <i class="fas fa-user icon"></i>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-bubble bot-message">
            <i class="fas fa-robot icon"></i>{message["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Show metadata if available
        if "metadata" in message:
            metadata = message["metadata"]
            st.markdown(f"""
            <div style="font-size: 0.8em; color: #6b7280; margin-top: 0.5rem;">
                <i class="fas fa-info-circle icon"></i>
                Sources: {metadata.get('doc_sources', 0)} documents
                {' | Web search used' if metadata.get('web_search_used') else ''}
            </div>
            """, unsafe_allow_html=True)

def handle_user_input(user_input: str):
    """Handle user input and generate response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            result = rag_pipeline.process_query(
                user_input, 
                response_mode=st.session_state.response_mode,
                use_web_search=True
            )
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"],
                "metadata": {
                    "doc_sources": result["doc_sources"],
                    "web_search_used": result["web_search_used"],
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message
            })
            logger.error(f"Error handling user input: {e}")
    
    st.rerun()

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <i class="fas fa-heart icon"></i>Built with Streamlit, Gemini AI, and SERPAPI
        <br>
        <i class="fas fa-code icon"></i>RAG Chatbot for Document Q&A
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render header
        render_header()
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            render_chat_interface()
        
        with col2:
            render_sidebar()
        
        # Render footer
        render_footer()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")

if __name__ == "__main__":
    main()