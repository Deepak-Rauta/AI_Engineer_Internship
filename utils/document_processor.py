import os
import tempfile
from pathlib import Path
import logging
import PyPDF2
from docx import Document
import chardet
from config.config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def process_uploaded_file(self, uploaded_file):
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension not in Config.ALLOWED_EXTENSIONS:
            raise ValueError(f"File type {file_extension} not supported")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if file_extension == '.pdf':
                text = self._extract_pdf(tmp_path)
            elif file_extension == '.docx':
                text = self._extract_docx(tmp_path)
            elif file_extension in ['.txt', '.md']:
                text = self._extract_text(tmp_path)
            else:
                raise ValueError(f"Unsupported file: {file_extension}")
            
            return text
        finally:
            os.unlink(tmp_path)
    
    def _extract_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_docx(self, file_path):
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_text(self, file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)['encoding']
        
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read().strip()
    
    def chunk_text(self, text):
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk)
                })
                
                # Start new chunk with some overlap
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap + " " + sentence
                chunk_id += 1
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })
        
        return chunks
    
    def _get_overlap(self, text):
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_text = text[-self.chunk_overlap:]
        sentence_start = overlap_text.find('. ')
        if sentence_start != -1:
            overlap_text = overlap_text[sentence_start + 2:]
        
        return overlap_text.strip()

def get_document_processor():
    return DocumentProcessor()
