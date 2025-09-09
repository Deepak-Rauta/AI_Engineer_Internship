import os
import tempfile
from typing import List, Dict, Any
import logging
from pathlib import Path
import streamlit as st

# Document processing imports
import PyPDF2
from docx import Document
import chardet

from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and chunk documents for RAG"""
    
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """
        Process uploaded file and extract text
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension not in Config.ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract text based on file type
                if file_extension == '.pdf':
                    text = self._extract_pdf_text(tmp_path)
                elif file_extension == '.docx':
                    text = self._extract_docx_text(tmp_path)
                elif file_extension in ['.txt', '.md']:
                    text = self._extract_text_file(tmp_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
                logger.info(f"Successfully processed {uploaded_file.name}")
                return text
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            # Detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text file: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            # Split text into sentences (simple approach)
            sentences = text.replace('\n', ' ').split('. ')
            
            chunks = []
            current_chunk = ""
            current_length = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence)
                
                # If adding this sentence would exceed chunk size, save current chunk
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': current_length
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                    chunk_id += 1
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
                    current_length = len(current_chunk)
            
            # Add final chunk if it exists
            if current_chunk.strip():
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
            
            logger.info(f"Created {len(chunks)} chunks from text")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        try:
            if len(text) <= self.chunk_overlap:
                return text
            
            # Get last chunk_overlap characters, but try to break at sentence boundary
            overlap_text = text[-self.chunk_overlap:]
            
            # Find the first sentence boundary in the overlap
            sentence_start = overlap_text.find('. ')
            if sentence_start != -1:
                overlap_text = overlap_text[sentence_start + 2:]
            
            return overlap_text.strip()
            
        except Exception as e:
            logger.error(f"Error getting overlap text: {str(e)}")
            return text[-self.chunk_overlap:] if len(text) > self.chunk_overlap else text

def get_document_processor() -> DocumentProcessor:
    """Get document processor instance"""
    return DocumentProcessor()
