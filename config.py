import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Base directories
    BASE_DIR = Path(__file__).parent
    UPLOADS_DIR = BASE_DIR / "uploads"
    VECTORSTORE_DIR = BASE_DIR / "vectorstore" 
    DATABASE_DIR = BASE_DIR / "database"
    EXPORTS_DIR = BASE_DIR / "exports"

    # LLM Provider: "groq" or "openai"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Model names
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Model configurations
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    
    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # FAISS settings
    FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"
    SIMILARITY_THRESHOLD = 0.8
    
    # Database
    DB_PATH = DATABASE_DIR / "qpg_database.db"
    
    # Question paper settings
    MARKS_DISTRIBUTION = {
        "PART_A": {"questions": 10, "marks_each": 2, "total": 20},
        "PART_B": {"questions": 5, "marks_each": 5, "total": 25}, 
        "PART_C": {"questions": 3, "marks_each": 10, "total": 30},
        "PART_D": {"questions": 1, "marks_each": 15, "total": 15}
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.UPLOADS_DIR, cls.VECTORSTORE_DIR, 
                        cls.DATABASE_DIR, cls.EXPORTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)