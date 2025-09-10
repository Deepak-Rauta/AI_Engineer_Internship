import streamlit as st
import logging
from config.config import Config
from models.llm import get_gemini_llm
from utils.rag_pipeline import get_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
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
</style>
""", unsafe_allow_html=True)

class ChatbotApp:
    def __init__(self):
        Config.validate_config()
        self.llm = get_gemini_llm()
        self.rag_pipeline = get_rag_pipeline()
        self._init_session_state()
    
    def _init_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'response_mode' not in st.session_state:
            st.session_state.response_mode = "detailed"
        if 'use_web_search' not in st.session_state:
            st.session_state.use_web_search = True
    
    def run(self):
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RAG Chatbot</h1>
            <p>AI Assistant with Document Knowledge & Web Search</p>
        </div>
        """, unsafe_allow_html=True)
        
        self._render_sidebar()
        self._render_chat()
    
    def _render_sidebar(self):
        with st.sidebar:
            st.markdown("## Settings")
            
            # Response mode
            response_mode = st.radio(
                "Response Style:",
                ["detailed", "concise"],
                index=0 if st.session_state.response_mode == "detailed" else 1
            )
            st.session_state.response_mode = response_mode
            
            # Web search toggle
            use_web_search = st.checkbox(
                "Enable web search",
                value=st.session_state.use_web_search
            )
            st.session_state.use_web_search = use_web_search
            
            # Document upload
            st.markdown("### Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['txt', 'pdf', 'docx', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process Documents"):
                    with st.spinner("Processing..."):
                        results = self.rag_pipeline.add_documents_from_files(uploaded_files)
                        
                        if results['total_chunks'] > 0:
                            st.success(f"Processed {results['total_chunks']} chunks")
                        
                        for error in results['errors']:
                            st.error(error)
            
            # Stats
            stats = self.rag_pipeline.get_pipeline_stats()
            num_docs = stats.get('vector_store', {}).get('num_documents', 0)
            st.info(f"Documents: {num_docs}")
            
            if num_docs > 0:
                if st.button("Clear Knowledge Base"):
                    self.rag_pipeline.clear_knowledge_base()
                    st.success("Cleared!")
                    st.rerun()
    
    def _render_chat(self):
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        st.text(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources = self._generate_response(prompt)
                        st.markdown(response)
                        
                        if sources and "No sources found" not in sources:
                            with st.expander("Sources"):
                                st.text(sources)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    def _generate_response(self, query):
        context, sources = self.rag_pipeline.retrieve_context(
            query, 
            use_web_search=st.session_state.use_web_search
        )
        
        response = self.llm.generate_response(
            prompt=query,
            context=context,
            response_mode=st.session_state.response_mode
        )
        
        return response, sources

def main():
    try:
        app = ChatbotApp()
        app.run()
    except Exception as e:
        st.error(f"App failed to start: {str(e)}")

if __name__ == "__main__":
    main()
