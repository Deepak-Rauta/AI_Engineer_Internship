"""
Document processing utilities
Handles file upload, text extraction, and chunking
"""
import os
import tempfile
from typing import List, Dict, Any
import pypdf

import docx
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx', '.md']
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from uploaded file
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Extracted text content
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.txt' or file_extension == '.md':
                return self._extract_from_text(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def _extract_from_text(self, file_path: str) -> str:
        """Extract text from text/markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        try:
            if len(text) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                
                # Try to break at sentence boundary
                if end < len(text):
                    # Look for sentence endings
                    for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                        if text[i] in '.!?':
                            end = i + 1
                            break
                
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                start = end - overlap
                
                if start >= len(text):
                    break
            
            logger.info(f"Created {len(chunks)} chunks from text")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [text]  # Return original text as fallback
    
    def process_uploaded_file(self, uploaded_file) -> List[str]:
        """
        Process uploaded file and return text chunks
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of text chunks
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract text
            text = self.extract_text_from_file(tmp_path)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            logger.info(f"Processed file: {uploaded_file.name}, created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise

# Global document processor instance
doc_processor = DocumentProcessor()