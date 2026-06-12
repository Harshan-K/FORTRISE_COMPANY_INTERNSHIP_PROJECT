import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import Config

class DatabaseManager:
    def __init__(self):
        Config.create_directories()
        self.db_path = Config.DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    chunks_count INTEGER DEFAULT 0
                )
            """)
            
            # Generated papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_name TEXT NOT NULL,
                    department TEXT,
                    exam_type TEXT,
                    duration TEXT,
                    total_marks INTEGER,
                    difficulty_level TEXT,
                    bloom_level TEXT,
                    questions_data TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    pdf_path TEXT
                )
            """)
            
            # Question history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_text TEXT NOT NULL,
                    question_type TEXT,
                    marks INTEGER,
                    difficulty TEXT,
                    bloom_level TEXT,
                    topic TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def add_document(self, filename: str, file_path: str, file_type: str) -> int:
        """Add document record to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, file_path, file_type)
                VALUES (?, ?, ?)
            """, (filename, file_path, file_type))
            return cursor.lastrowid
    
    def update_document_processed(self, doc_id: int, chunks_count: int):
        """Update document processing status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents 
                SET processed = TRUE, chunks_count = ?
                WHERE id = ?
            """, (chunks_count, doc_id))
    
    def save_generated_paper(self, paper_data: Dict[str, Any]) -> int:
        """Save generated paper to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO generated_papers 
                (subject_name, department, exam_type, duration, total_marks,
                 difficulty_level, bloom_level, questions_data, pdf_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_data.get('subject') or paper_data.get('subject_name'),
                paper_data.get('department'),
                paper_data.get('exam_type'),
                paper_data.get('duration'),
                paper_data.get('total_marks'),
                paper_data.get('difficulty') or paper_data.get('difficulty_level'),
                paper_data.get('bloom_level'),
                json.dumps(paper_data.get('questions', [])),
                str(paper_data.get('pdf_path') or '')
            ))
            return cursor.lastrowid
    
    def save_question(self, question_data: Dict[str, Any]):
        """Save individual question to history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO question_history 
                (question_text, question_type, marks, difficulty, bloom_level, topic)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                question_data.get('question'),
                question_data.get('type'),
                question_data.get('marks'),
                question_data.get('difficulty'),
                question_data.get('bloom_level'),
                question_data.get('topic')
            ))
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM documents WHERE processed = TRUE")
            total_docs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM question_history")
            total_questions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM generated_papers")
            total_papers = cursor.fetchone()[0]

            cursor.execute("""
                SELECT difficulty, COUNT(*)
                FROM question_history
                GROUP BY difficulty
            """)
            difficulty_dist = dict(cursor.fetchall())

            cursor.execute("""
                SELECT bloom_level, COUNT(*)
                FROM question_history
                GROUP BY bloom_level
            """)
            bloom_dist = dict(cursor.fetchall())

            cursor.execute("""
                SELECT subject_name, exam_type, total_marks, generated_at
                FROM generated_papers
                ORDER BY generated_at DESC
                LIMIT 5
            """)
            recent_papers = cursor.fetchall()

            return {
                "total_documents": total_docs,
                "total_questions": total_questions,
                "total_papers": total_papers,
                "difficulty_distribution": difficulty_dist,
                "bloom_distribution": bloom_dist,
                "recent_papers": recent_papers
            }