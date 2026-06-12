import os
import re
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep academic content
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        
        # Remove very short lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(lines)
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type"""
        extractors = {
            'pdf': self.extract_text_from_pdf,
            'docx': self.extract_text_from_docx,
            'txt': self.extract_text_from_txt
        }
        
        if file_type.lower() not in extractors:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        text = extractors[file_type.lower()](file_path)
        return self.clean_text(text)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                'content': chunk,
                'chunk_id': i,
                'length': len(chunk),
                'metadata': {
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return chunked_docs
    
    def process_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Complete document processing pipeline"""
        file_type = Path(filename).suffix[1:].lower()
        
        # Extract text
        raw_text = self.extract_text(file_path, file_type)
        
        # Chunk text
        chunks = self.chunk_text(raw_text)
        
        return {
            'filename': filename,
            'file_type': file_type,
            'raw_text': raw_text,
            'chunks': chunks,
            'chunks_count': len(chunks),
            'total_length': len(raw_text)
        }