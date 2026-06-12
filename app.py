import gradio as gr
import os
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

# Import project modules
from config import Config
from database.db_manager import DatabaseManager
from utils.document_processor import DocumentProcessor
from vectorstore.faiss_store import VectorStore
from rag.rag_pipeline import RAGPipeline
from agents.crew_agents import QuestionGenerationAgents
from workflows.langgraph_workflow import LangGraphWorkflow
from utils.pdf_generator import PDFGenerator

class QuestionPaperGeneratorApp:
    def __init__(self):
        """Initialize the application"""
        Config.create_directories()
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline()
        self.crew_agents = QuestionGenerationAgents(self.rag_pipeline)
        self.workflow = LangGraphWorkflow(self.rag_pipeline, self.vector_store)
        self.pdf_generator = PDFGenerator()
        
        # Application state
        self.uploaded_files = []
        self.processed_documents = 0
        
    def upload_documents(self, files) -> str:
        """Handle document upload and processing"""
        if not files:
            return "No files uploaded!"

        # Normalize: Gradio may pass a single file or a list
        if not isinstance(files, list):
            files = [files]

        results = []
        processed_count = 0

        import shutil
        try:
            for file in files:
                # Gradio 4.x passes NamedString / filepath string — extract path safely
                if isinstance(file, str):
                    temp_path = file
                elif hasattr(file, 'name'):
                    temp_path = file.name
                else:
                    # last resort: write bytes to temp file
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(bytes(file))
                    tmp.close()
                    temp_path = tmp.name

                filename = Path(temp_path).name
                file_path = Config.UPLOADS_DIR / filename
                shutil.copy2(temp_path, str(file_path))

                # Process document
                processed_doc = self.doc_processor.process_document(str(file_path), filename)
                
                # Add to vector store
                chunks_added = self.vector_store.add_documents(
                    processed_doc['chunks'],
                    filename
                )
                
                # Save to database
                doc_id = self.db_manager.add_document(
                    filename, str(file_path), processed_doc['file_type']
                )
                self.db_manager.update_document_processed(doc_id, chunks_added)
                
                results.append(f"[OK] {filename}: {chunks_added} chunks processed")
                processed_count += 1
            
            self.processed_documents += processed_count
            
            return f"Successfully processed {processed_count} documents:\n" + "\n".join(results)
            
        except Exception as e:
            import traceback
            return f"Error processing documents: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
    
    def generate_questions(self, subject: str, department: str, exam_type: str, 
                          duration: str, total_marks: int, difficulty: str, 
                          bloom_level: str, use_crew: bool, use_workflow: bool) -> Tuple[str, str, str]:
        """Generate questions based on requirements"""
        
        if not subject:
            return "Please enter a subject name!", "", None
        
        try:
            # Prepare requirements
            requirements = {
                'subject': subject,
                'department': department,
                'exam_type': exam_type,
                'duration': duration,
                'total_marks': total_marks,
                'difficulty': difficulty,
                'bloom_level': bloom_level
            }
            
            # Choose generation method
            if use_workflow:
                result = self.workflow.execute_workflow(requirements)
                questions = result['questions']
                formatted_paper = result['formatted_paper']
                pdf_path = result.get('pdf_path', '') or None

            elif use_crew:
                result = self.crew_agents.execute_crew(requirements)
                questions = result['questions']
                formatted_paper = self._format_questions_simple(questions, requirements)
                pdf_path = self.pdf_generator.generate_question_paper_pdf(formatted_paper) or None

            else:
                # Direct RAG generation
                questions = self.rag_pipeline.generate_questions_batch(requirements)
                formatted_paper = self._format_questions_simple(questions, requirements)
                pdf_path = self.pdf_generator.generate_question_paper_pdf(formatted_paper) or None

            # Validate pdf_path is an actual file
            if pdf_path and not os.path.isfile(pdf_path):
                pdf_path = None
            
            # Save to database
            paper_data = {**requirements, 'questions': questions, 'pdf_path': pdf_path}
            self.db_manager.save_generated_paper(paper_data)
            
            # Save individual questions
            for question in questions:
                self.db_manager.save_question(question)
            
            # Format output
            questions_text = self._format_questions_display(questions)
            analysis_text = self._generate_analysis(questions, requirements)
            
            return questions_text, analysis_text, pdf_path
            
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}", "", None
    
    def _format_questions_simple(self, questions: List[Dict[str, Any]], 
                                requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Format questions into paper structure"""
        
        # Group questions by part
        parts = {'PART_A': [], 'PART_B': [], 'PART_C': [], 'PART_D': []}
        
        for question in questions:
            part = question.get('part', 'PART_A')
            if part in parts:
                parts[part].append(question)
        
        return {
            'header': {
                'college_name': 'University Name',
                'department': requirements.get('department', 'Computer Science'),
                'subject': requirements.get('subject', 'Subject Name'),
                'exam_type': requirements.get('exam_type', 'Mid-Term'),
                'duration': requirements.get('duration', '3 Hours'),
                'total_marks': requirements.get('total_marks', 90)
            },
            'instructions': [
                'Answer all questions.',
                'Figures to the right indicate full marks.',
                'Draw diagrams wherever necessary.'
            ],
            'parts': parts,
            'total_questions': len(questions)
        }
    
    def _format_questions_display(self, questions: List[Dict[str, Any]]) -> str:
        """Format questions for display in UI"""
        if not questions:
            return "No questions generated."
        
        output = []
        current_part = None
        question_num = 1
        
        for question in questions:
            part = question.get('part', 'PART_A')
            
            if part != current_part:
                if current_part is not None:
                    output.append("")
                output.append(f"\n{part}:")
                output.append("-" * 50)
                current_part = part
                question_num = 1
            
            marks = question.get('marks', 2)
            q_text = question.get('question', 'Question text missing')
            output.append(f"{question_num}. {q_text} ({marks} marks)")
            output.append("")
            question_num += 1
        
        return "\n".join(output)
    
    def _generate_analysis(self, questions: List[Dict[str, Any]], 
                          requirements: Dict[str, Any]) -> str:
        """Generate analysis of generated questions"""
        
        if not questions:
            return "No questions to analyze."
        
        # Count by parts
        parts_count = {}
        difficulty_count = {}
        bloom_count = {}
        total_marks = 0
        
        for question in questions:
            part = question.get('part', 'PART_A')
            difficulty = question.get('difficulty', 'Medium')
            bloom = question.get('bloom_level', 'Apply')
            marks = question.get('marks', 2)
            
            parts_count[part] = parts_count.get(part, 0) + 1
            difficulty_count[difficulty] = difficulty_count.get(difficulty, 0) + 1
            bloom_count[bloom] = bloom_count.get(bloom, 0) + 1
            total_marks += marks
        
        analysis = [
            "QUESTION PAPER ANALYSIS",
            "=" * 30,
            f"Total Questions: {len(questions)}",
            f"Total Marks: {total_marks}",
            "",
            "Distribution by Parts:",
            *[f"  {part}: {count} questions" for part, count in parts_count.items()],
            "",
            "Difficulty Distribution:",
            *[f"  {diff}: {count} questions" for diff, count in difficulty_count.items()],
            "",
            "Bloom's Taxonomy Distribution:",
            *[f"  {bloom}: {count} questions" for bloom, count in bloom_count.items()],
        ]
        
        return "\n".join(analysis)
    
    def get_analytics_data(self) -> Tuple[str, Any, Any, Any, Any]:
        """Get analytics data for dashboard"""
        try:
            analytics = self.db_manager.get_analytics_data()
            vector_stats = self.vector_store.get_stats()

            recent = analytics.get('recent_papers', [])
            recent_str = "\n".join(
                f"  - {r[0]} | {r[1]} | {r[2]} marks | {r[3][:16]}"
                for r in recent
            ) if recent else "  None yet."

            summary = f"""DATABASE ANALYTICS
==================
Total Documents Processed : {analytics['total_documents']}
Total Questions Generated  : {analytics['total_questions']}
Total Papers Generated     : {analytics['total_papers']}

VECTOR STORE STATISTICS
=======================
Total Document Chunks  : {vector_stats['total_documents']}
FAISS Index Size       : {vector_stats['index_size']}
Embedding Dimension    : {vector_stats['embedding_dimension']}

RECENT PAPERS
=============
{recent_str}

DIFFICULTY DISTRIBUTION
=======================
""" + ("\n".join(f"  {k}: {v} questions" for k, v in analytics.get('difficulty_distribution', {}).items()) or "  No data yet.")

            # Chart 1: Overview bar chart
            fig1 = go.Figure(go.Bar(
                x=['Documents', 'Questions', 'Papers'],
                y=[analytics['total_documents'], analytics['total_questions'], analytics['total_papers']],
                marker=dict(
                    color=['#c4a050', '#8b6914', '#d4b464'],
                    line=dict(color='rgba(196,160,80,0.3)', width=1)
                ),
                text=[analytics['total_documents'], analytics['total_questions'], analytics['total_papers']],
                textposition='auto',
                textfont=dict(color='#0a0f1e', size=14, family='Inter')
            ))
            fig1.update_layout(
                title=dict(text='Overall Statistics', font=dict(color='#c4a050', size=16, family='Playfair Display')),
                plot_bgcolor='rgba(5,10,30,0.8)', paper_bgcolor='rgba(8,15,40,0.9)',
                font=dict(color='#d4c8a8', family='Inter'),
                xaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8'),
                yaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8'),
                height=350, margin=dict(l=40, r=40, t=50, b=40)
            )

            # Chart 2: Difficulty pie chart
            diff_dist = analytics.get('difficulty_distribution', {})
            if diff_dist:
                fig2 = go.Figure(go.Pie(
                    labels=list(diff_dist.keys()),
                    values=list(diff_dist.values()),
                    hole=0.45,
                    marker=dict(colors=['#c4a050','#8b6914','#d4b464','#6b4e10'],
                                line=dict(color='rgba(5,10,30,0.8)', width=2)),
                    textfont=dict(color='#e8e0d0', size=12)
                ))
            else:
                fig2 = go.Figure(go.Pie(
                    labels=['No Data'], values=[1], hole=0.45,
                    marker=dict(colors=['rgba(196,160,80,0.2)'])
                ))
            fig2.update_layout(
                title=dict(text='Difficulty Distribution', font=dict(color='#c4a050', size=16, family='Playfair Display')),
                paper_bgcolor='rgba(8,15,40,0.9)',
                font=dict(color='#d4c8a8', family='Inter'),
                legend=dict(font=dict(color='#d4c8a8')),
                height=350, margin=dict(l=40, r=40, t=50, b=40)
            )

            # Chart 3: Bloom's taxonomy bar chart
            bloom_labels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            bloom_dist   = analytics.get('bloom_distribution', {})
            bloom_values = [bloom_dist.get(l, 0) for l in bloom_labels]
            fig3 = go.Figure(go.Bar(
                x=bloom_labels, y=bloom_values,
                marker=dict(
                    color=bloom_values,
                    colorscale=[[0,'#3a2800'],[0.5,'#8b6914'],[1,'#c4a050']],
                    line=dict(color='rgba(196,160,80,0.3)', width=1)
                ),
                text=bloom_values, textposition='auto',
                textfont=dict(color='#0a0f1e', size=13)
            ))
            fig3.update_layout(
                title=dict(text="Bloom's Taxonomy Distribution", font=dict(color='#c4a050', size=16, family='Playfair Display')),
                plot_bgcolor='rgba(5,10,30,0.8)', paper_bgcolor='rgba(8,15,40,0.9)',
                font=dict(color='#d4c8a8', family='Inter'),
                xaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8'),
                yaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8'),
                height=350, margin=dict(l=40, r=40, t=50, b=40)
            )

            # Chart 4: Papers over time line chart
            recent_papers = analytics.get('recent_papers', [])
            if recent_papers:
                dates      = [r[3][:10] for r in recent_papers][::-1]
                marks_list = [r[2]      for r in recent_papers][::-1]
            else:
                dates      = ['No Data']
                marks_list = [0]
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=dates, y=marks_list,
                mode='lines+markers',
                line=dict(color='#c4a050', width=2.5),
                marker=dict(size=9, color='#c4a050',
                            line=dict(color='rgba(196,160,80,0.3)', width=2)),
                fill='tozeroy',
                fillcolor='rgba(196,160,80,0.08)',
                name='Marks'
            ))
            fig4.update_layout(
                title=dict(text='Recent Papers — Marks Over Time', font=dict(color='#c4a050', size=16, family='Playfair Display')),
                plot_bgcolor='rgba(5,10,30,0.8)', paper_bgcolor='rgba(8,15,40,0.9)',
                font=dict(color='#d4c8a8', family='Inter'),
                xaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8', title='Date'),
                yaxis=dict(gridcolor='rgba(196,160,80,0.1)', color='#d4c8a8', title='Total Marks'),
                height=350, margin=dict(l=40, r=40, t=50, b=40)
            )

            return summary, fig1, fig2, fig3, fig4

        except Exception as e:
            import traceback
            empty = go.Figure()
            return f"Error loading analytics: {str(e)}\n{traceback.format_exc()}", empty, empty, empty, empty
    
    def create_gradio_interface(self):
        """Create the Gradio interface"""

        css = """
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global reset & background ── */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body, .gradio-container, #root {
            background:
                linear-gradient(160deg, rgba(10,24,60,0.82) 0%, rgba(20,40,90,0.78) 40%, rgba(10,24,60,0.85) 100%),
                url('https://images.unsplash.com/photo-1523050854058-8df90110c9f1?w=1920&q=90') center/cover no-repeat fixed !important;
            min-height: 100vh;
            font-family: 'Inter', sans-serif !important;
            color: #f0ede8 !important;
        }

        /* ── Container ── */
        .gradio-container {
            max-width: 1380px !important;
            margin: 0 auto !important;
            padding: 0 28px 56px !important;
            background: transparent !important;
        }

        /* ── Hero Header ── */
        .hero-header { margin-bottom: 28px; border-radius: 20px; overflow: hidden; }
        .hero-bg {
            position: relative;
            background:
                linear-gradient(170deg, rgba(8,18,50,0.45) 0%, rgba(14,30,80,0.60) 100%),
                url('https://images.unsplash.com/photo-1497633762265-9d179a990aa6?w=1800&q=85') center 30%/cover no-repeat;
            border-radius: 20px;
            padding: 64px 48px 54px;
            border: 1px solid rgba(210,180,100,0.35);
            box-shadow: 0 12px 60px rgba(0,0,0,0.55), inset 0 1px 0 rgba(210,180,100,0.25);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(210,180,100,0.15);
            border: 1px solid rgba(210,180,100,0.55);
            color: #e8c96a;
            font-size: 11px; font-weight: 700;
            letter-spacing: 3.5px; text-transform: uppercase;
            padding: 6px 22px; border-radius: 30px; margin-bottom: 20px;
        }
        .hero-title {
            font-family: 'Playfair Display', serif !important;
            font-size: 56px; font-weight: 700;
            color: #ffffff; line-height: 1.12; margin-bottom: 14px;
            text-shadow: 0 3px 24px rgba(0,0,0,0.5);
        }
        .hero-title span { color: #e8c96a; }
        .hero-subtitle {
            font-size: 17px; color: rgba(240,235,220,0.88);
            font-weight: 300; letter-spacing: 0.4px;
            margin: 0 auto 30px; max-width: 640px; line-height: 1.7;
        }
        .hero-pills {
            display: flex; justify-content: center;
            flex-wrap: wrap; gap: 10px;
        }
        .hero-pill {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(210,180,100,0.38);
            color: #f0d88a; font-size: 12px; font-weight: 500;
            padding: 6px 16px; border-radius: 20px; letter-spacing: 0.4px;
            backdrop-filter: blur(4px);
        }

        /* ── Stat cards ── */
        .stat-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.10) 0%, rgba(14,30,80,0.70) 100%);
            border: 1px solid rgba(210,180,100,0.28);
            border-radius: 16px; padding: 26px 20px;
            text-align: center; backdrop-filter: blur(14px);
            transition: transform 0.25s, box-shadow 0.25s;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }
        .stat-card:hover { transform: translateY(-4px); box-shadow: 0 14px 36px rgba(0,0,0,0.38); }
        .stat-number { font-size: 16px; font-weight: 700; color: #e8c96a; line-height: 1.3; letter-spacing: 0.3px; margin-bottom: 4px; }
        .stat-label  { font-size: 11px; color: rgba(240,235,220,0.60); letter-spacing: 1.2px; text-transform: uppercase; }
        .stat-card:hover .stat-number { color: #f5d97a; }

        /* ── Section title ── */
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 18px; font-weight: 600; color: #e8c96a;
            margin-bottom: 16px; padding-bottom: 10px;
            border-bottom: 1px solid rgba(210,180,100,0.25);
            letter-spacing: 0.3px;
        }

        /* ── Upload banner ── */
        .upload-banner {
            background:
                linear-gradient(170deg, rgba(8,18,50,0.52) 0%, rgba(14,30,80,0.68) 100%),
                url('https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=900&q=85') center/cover no-repeat;
            border-radius: 14px; padding: 36px 28px;
            text-align: center;
            border: 1.5px dashed rgba(210,180,100,0.40);
            margin-bottom: 18px;
        }
        .upload-banner-title { font-family:'Playfair Display',serif; font-size:20px; font-weight:600; color:#e8c96a; margin-bottom:6px; }
        .upload-banner-sub   { font-size:13px; color:rgba(240,235,220,0.65); }

        /* ── Feature pills ── */
        .feature-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
        .feature-pill {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(210,180,100,0.25);
            border-radius: 8px; padding: 7px 14px;
            font-size: 12px; font-weight: 500; color: #e8c96a;
        }

        /* ── PDF download file component ── */
        #pdf-download {
            background: linear-gradient(135deg, rgba(212,168,50,0.12) 0%, rgba(160,120,24,0.10) 100%) !important;
            border: 1px solid rgba(210,180,100,0.45) !important;
            border-radius: 12px !important;
            padding: 4px !important;
            margin-top: 8px !important;
        }
        #pdf-download .download-link {
            color: #e8c96a !important;
            font-weight: 600 !important;
            font-size: 14px !important;
        }
        #pdf-download button {
            background: linear-gradient(135deg, #d4a832 0%, #a07818 100%) !important;
            color: #0a0f1e !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            letter-spacing: 0.5px !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 15px rgba(210,168,50,0.35) !important;
        }
        #pdf-download button:hover {
            background: linear-gradient(135deg, #e0b840 0%, #b08828 100%) !important;
            box-shadow: 0 6px 22px rgba(210,168,50,0.52) !important;
        }
        .panel-box {
            background: rgba(10,22,58,0.88) !important;
            border: 1px solid rgba(210,180,100,0.14) !important;
            border-radius: 16px !important;
            padding: 32px !important;
            margin-bottom: 8px !important;
        }

        /* ── Inputs ── */
        label, .label-wrap span, span.svelte-1gfknih {
            color: rgba(240,235,220,0.82) !important;
            font-size: 13px !important; font-weight: 500 !important;
            letter-spacing: 0.3px !important;
        }
        input, textarea, select {
            background: rgba(8,18,52,0.75) !important;
            border: 1px solid rgba(210,180,100,0.28) !important;
            border-radius: 10px !important;
            color: #f0ede8 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            transition: border-color 0.2s, box-shadow 0.2s !important;
        }
        input:focus, textarea:focus {
            border-color: rgba(210,180,100,0.65) !important;
            box-shadow: 0 0 0 3px rgba(210,180,100,0.12) !important;
            outline: none !important;
        }
        .gr-form, .gr-box { background: transparent !important; border: none !important; }
        /* Dropdown options */
        .gr-dropdown ul, ul[role=listbox] {
            background: rgba(12,26,68,0.98) !important;
            border: 1px solid rgba(210,180,100,0.25) !important;
            border-radius: 10px !important;
        }
        ul[role=listbox] li { color: #f0ede8 !important; }
        ul[role=listbox] li:hover, ul[role=listbox] li.selected { background: rgba(210,180,100,0.14) !important; }

        /* ── Buttons ── */
        button.primary {
            background: linear-gradient(135deg, #d4a832 0%, #a07818 100%) !important;
            color: #0a0f1e !important; border: none !important;
            border-radius: 10px !important; font-weight: 700 !important;
            font-size: 14px !important; letter-spacing: 0.8px !important;
            text-transform: uppercase !important;
            padding: 13px 32px !important;
            box-shadow: 0 4px 18px rgba(210,168,50,0.35) !important;
            transition: all 0.25s !important;
        }
        button.primary:hover {
            background: linear-gradient(135deg, #e0b840 0%, #b08828 100%) !important;
            box-shadow: 0 8px 28px rgba(210,168,50,0.52) !important;
            transform: translateY(-2px) !important;
        }
        button.secondary {
            background: rgba(255,255,255,0.07) !important;
            color: #e8c96a !important;
            border: 1px solid rgba(210,180,100,0.38) !important;
            border-radius: 10px !important; font-weight: 600 !important;
            font-size: 13px !important; letter-spacing: 0.5px !important;
            padding: 11px 26px !important; transition: all 0.2s !important;
        }
        button.secondary:hover {
            background: rgba(210,180,100,0.14) !important;
            border-color: rgba(210,180,100,0.6) !important;
        }

        /* ── Slider ── */
        input[type=range] { accent-color: #d4a832 !important; }

        /* ── Checkbox ── */
        input[type=checkbox] { accent-color: #d4a832 !important; width:15px; height:15px; }

        /* ── Output textbox ── */
        .output-textbox textarea {
            background: rgba(4,10,32,0.82) !important;
            border: 1px solid rgba(210,180,100,0.22) !important;
            color: #ddd8c4 !important;
            font-family: 'Courier New', monospace !important;
            font-size: 13px !important; line-height: 1.65 !important;
        }

        /* ── File upload ── */
        .gr-file-upload, [data-testid='file-upload'] {
            background: rgba(8,18,52,0.65) !important;
            border: 2px dashed rgba(210,180,100,0.32) !important;
            border-radius: 12px !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: rgba(8,18,52,0.4); }
        ::-webkit-scrollbar-thumb { background: rgba(210,180,100,0.45); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(210,180,100,0.7); }

        /* ── Divider ── */
        .gold-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(210,180,100,0.45), transparent);
            margin: 22px 0;
        }

        /* ── Tab banner (shared) ── */
        .tab-banner {
            border-radius: 14px;
            padding: 30px 36px;
            border: 1px solid rgba(210,180,100,0.22);
            margin-bottom: 26px;
        }
        .tab-banner-title {
            font-family: 'Playfair Display', serif;
            font-size: 26px; font-weight: 700; color: #e8c96a; margin-bottom: 6px;
        }
        .tab-banner-sub { font-size: 14px; color: rgba(240,235,220,0.65); line-height: 1.6; }

        /* ── Config items ── */
        .config-item {
            background: rgba(8,18,52,0.72);
            border: 1px solid rgba(210,180,100,0.20);
            border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
        }
        .config-key   { font-size: 10px; color: #e8c96a; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; }
        .config-value { font-size: 13px; color: #f0ede8; font-family: 'Courier New', monospace; word-break: break-all; }

        /* ── Plots ── */
        .gr-plot { border-radius: 12px !important; overflow: hidden !important; }

        /* ── Row / column flex fix ── */
        .gradio-row { display: flex !important; flex-wrap: wrap !important; gap: 20px !important; align-items: flex-start !important; }
        .gradio-column { display: flex !important; flex-direction: column !important; min-width: 0 !important; }
        """

        with gr.Blocks(css=css, title="QuestionHub — AI Paper Generator") as demo:

            # ── HERO ─────────────────────────────────────────────────────
            gr.HTML("""
            <div class="hero-header">
              <div class="hero-bg" style="text-align:center;">
                <div class="hero-badge">Powered by RAG &middot; FAISS &middot; LLM &middot; LangGraph</div>
                <div class="hero-title">Question<span>Hub</span> AI</div>
                <div class="hero-subtitle">
                  Generate university-level examination papers with intelligent retrieval,
                  Bloom&rsquo;s taxonomy alignment and multi-agent orchestration.
                </div>
                <div class="hero-pills">
                  <span class="hero-pill">PDF / DOCX / TXT Upload</span>
                  <span class="hero-pill">RAG Pipeline</span>
                  <span class="hero-pill">FAISS Vector Search</span>
                  <span class="hero-pill">LangGraph Workflow</span>
                  <span class="hero-pill">Analytics Dashboard</span>
                  <span class="hero-pill">PDF Export</span>
                </div>
              </div>
            </div>
            """)

            # ── NAV CARDS ────────────────────────────────────────────────
            with gr.Row(elem_id="nav-cards-row"):
                btn_upload   = gr.Button(elem_id="nav-upload",   elem_classes="nav-card")
                btn_generate = gr.Button(elem_id="nav-generate", elem_classes="nav-card")
                btn_analytics= gr.Button(elem_id="nav-analytics",elem_classes="nav-card")
                btn_settings = gr.Button(elem_id="nav-settings", elem_classes="nav-card")

            gr.HTML("""
            <style>
              /* Override Gradio button inside nav card */
              #nav-upload, #nav-generate, #nav-analytics, #nav-settings {
                background: linear-gradient(145deg,rgba(255,255,255,0.10) 0%,rgba(14,30,80,0.70) 100%) !important;
                border: 1px solid rgba(210,180,100,0.28) !important;
                border-radius: 16px !important;
                padding: 0 !important;
                overflow: hidden !important;
                cursor: pointer !important;
                transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.25) !important;
                flex: 1 1 0 !important;
                min-width: 0 !important;
                color: transparent !important;
                font-size: 0 !important;
                height: auto !important;
              }
              #nav-upload::before    { content: ''; display:block; width:100%; height:100px; background: url("https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=400&q=80") center/cover no-repeat; border-radius:14px 14px 0 0; }
              #nav-generate::before  { content: ''; display:block; width:100%; height:100px; background: url("https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=400&q=80") center/cover no-repeat; border-radius:14px 14px 0 0; }
              #nav-analytics::before { content: ''; display:block; width:100%; height:100px; background: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&q=80") center/cover no-repeat; border-radius:14px 14px 0 0; }
              #nav-settings::before  { content: ''; display:block; width:100%; height:100px; background: url("https://images.unsplash.com/photo-1518770660439-4636190af475?w=400&q=80") center/cover no-repeat; border-radius:14px 14px 0 0; }
              #nav-upload::after    { content: 'Document Upload\A Index your academic files';    white-space:pre; display:block; padding:14px 16px; font-size:15px; font-weight:700; color:#e8c96a; font-family:'Inter',sans-serif; text-align:center; line-height:1.4; }
              #nav-generate::after  { content: 'Generate Questions\A AI-powered paper creation';  white-space:pre; display:block; padding:14px 16px; font-size:15px; font-weight:700; color:#e8c96a; font-family:'Inter',sans-serif; text-align:center; line-height:1.4; }
              #nav-analytics::after { content: 'Analytics\A Statistics & visualizations';         white-space:pre; display:block; padding:14px 16px; font-size:15px; font-weight:700; color:#e8c96a; font-family:'Inter',sans-serif; text-align:center; line-height:1.4; }
              #nav-settings::after  { content: 'Settings\A Model & system config';                white-space:pre; display:block; padding:14px 16px; font-size:15px; font-weight:700; color:#e8c96a; font-family:'Inter',sans-serif; text-align:center; line-height:1.4; }
              #nav-upload:hover, #nav-generate:hover, #nav-analytics:hover, #nav-settings:hover {
                transform: translateY(-5px) !important;
                box-shadow: 0 16px 40px rgba(0,0,0,0.45) !important;
                border-color: rgba(210,180,100,0.55) !important;
              }
              .nav-card-active {
                border-color: rgba(210,180,100,0.80) !important;
                box-shadow: 0 0 0 2px rgba(210,180,100,0.40), 0 14px 36px rgba(0,0,0,0.4) !important;
                transform: translateY(-5px) !important;
              }
              .nav-card-active::after { color: #f5d97a !important; }
              #nav-cards-row { display:flex !important; gap:18px !important; margin-bottom:28px !important; flex-wrap:nowrap !important; }
            </style>
            """)

            # ── PANELS (no gr.Tabs) ──────────────────────────────────────────

            # ── PANEL 1 : DOCUMENT UPLOAD ───────────────────────────────
            with gr.Column(visible=True, elem_classes="panel-box") as panel_upload:
                gr.HTML("""
                <div class="tab-banner" style="background:
                    linear-gradient(170deg,rgba(8,18,50,0.50) 0%,rgba(14,30,80,0.65) 100%),
                    url('https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=1400&q=85') center 40%/cover no-repeat;">
                  <div class="tab-banner-title">Knowledge Base Builder</div>
                  <div class="tab-banner-sub">Upload syllabi, textbooks, lecture notes and past papers to power AI question generation.</div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML("""
                        <div class="upload-banner">
                          <div class="upload-banner-title">Drop Academic Documents Here</div>
                          <div class="upload-banner-sub">Supports PDF, DOCX and TXT &middot; Multiple files allowed</div>
                        </div>
                        """)
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt"],
                            label="Select Documents"
                        )
                        upload_btn = gr.Button("Process and Index Documents", variant="primary", size="lg")
                    with gr.Column(scale=7):
                        gr.HTML("""
                        <div class="feature-row">
                          <span class="feature-pill">Auto Chunking</span>
                          <span class="feature-pill">Embedding Generation</span>
                          <span class="feature-pill">FAISS Indexing</span>
                          <span class="feature-pill">Database Storage</span>
                        </div>
                        """)
                        upload_status = gr.Textbox(
                            label="Processing Log", lines=16,
                            placeholder="Processing results will appear here after upload...",
                            elem_classes="output-textbox"
                        )

            # ── PANEL 2 : GENERATE QUESTIONS ────────────────────────────
            with gr.Column(visible=False, elem_classes="panel-box") as panel_generate:
                gr.HTML("""
                <div class="tab-banner" style="background:
                    linear-gradient(170deg,rgba(8,18,50,0.50) 0%,rgba(14,30,80,0.65) 100%),
                    url('https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=1400&q=85') center 35%/cover no-repeat;">
                  <div class="tab-banner-title">Intelligent Question Generator</div>
                  <div class="tab-banner-sub">Configure exam parameters and let the AI craft a complete, Bloom&rsquo;s-aligned question paper from your knowledge base.</div>
                </div>
                """)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        gr.HTML('<div class="section-title">Exam Configuration</div>')
                        subject = gr.Textbox(label="Subject Name", placeholder="e.g., Data Structures and Algorithms")
                        department = gr.Textbox(label="Department", value="Computer Science")
                        with gr.Row():
                            exam_type = gr.Dropdown(choices=["Mid-Term","Final","Quiz","Assignment"], value="Mid-Term", label="Exam Type")
                            duration  = gr.Dropdown(choices=["1 Hour","2 Hours","3 Hours","4 Hours"], value="3 Hours", label="Duration")
                        total_marks = gr.Slider(minimum=50, maximum=150, value=90, step=5, label="Total Marks")
                        with gr.Row():
                            difficulty  = gr.Dropdown(choices=["Easy","Medium","Hard","Mixed"], value="Medium", label="Difficulty Level")
                            bloom_level = gr.Dropdown(choices=["Remember","Understand","Apply","Analyze","Evaluate","Create"], value="Apply", label="Bloom's Level")
                        gr.HTML('<div class="gold-divider"></div>')
                        gr.HTML('<div class="section-title">Generation Engine</div>')
                        use_crew     = gr.Checkbox(label="Use CrewAI Multi-Agent System", value=False)
                        use_workflow = gr.Checkbox(label="Use LangGraph Workflow", value=True)
                        gr.HTML('<div style="margin-top:24px;"></div>')
                        generate_btn = gr.Button("Generate Question Paper", variant="primary", size="lg")
                    with gr.Column(scale=6):
                        gr.HTML('<div class="section-title">Generated Question Paper</div>')
                        questions_output = gr.Textbox(label="Question Paper", lines=22, max_lines=30, elem_classes="output-textbox")
                        gr.HTML('<div class="gold-divider"></div>')
                        gr.HTML('<div class="section-title">Paper Analysis</div>')
                        analysis_output = gr.Textbox(label="Analysis Report", lines=10, elem_classes="output-textbox")
                        gr.HTML('<div class="gold-divider"></div>')
                        gr.HTML('<div class="section-title">Export</div>')
                        download_btn = gr.HTML('<div style="color:rgba(240,235,220,0.4);font-size:13px;padding:12px 0;">Generate a paper first to enable download.</div>')
                        pdf_file_out = gr.File(
                            label="Download Question Paper PDF",
                            visible=False,
                            elem_id="pdf-download"
                        )

            # ── PANEL 3 : ANALYTICS ─────────────────────────────────────
            with gr.Column(visible=False, elem_classes="panel-box") as panel_analytics:
                gr.HTML("""
                <div class="tab-banner" style="background:
                    linear-gradient(170deg,rgba(8,18,50,0.52) 0%,rgba(14,30,80,0.68) 100%),
                    url('https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1400&q=85') center/cover no-repeat;">
                  <div class="tab-banner-title">Analytics Dashboard</div>
                  <div class="tab-banner-sub">Live statistics from your question generation sessions, vector store and database.</div>
                </div>
                """)
                analytics_btn = gr.Button("Refresh Analytics", variant="secondary")
                gr.HTML('<div style="margin-bottom:18px;"></div>')
                analytics_text = gr.Textbox(label="Analytics Summary", lines=20, interactive=False, elem_classes="output-textbox")
                gr.HTML('<div class="gold-divider"></div>')
                gr.HTML('<div class="section-title" style="margin-bottom:18px;">Visual Analytics</div>')
                with gr.Row():
                    chart_overview   = gr.Plot(label="Overall Statistics")
                    chart_difficulty = gr.Plot(label="Difficulty Distribution")
                with gr.Row():
                    chart_bloom    = gr.Plot(label="Bloom's Taxonomy")
                    chart_timeline = gr.Plot(label="Papers Over Time")

            # ── PANEL 4 : SETTINGS ──────────────────────────────────────
            with gr.Column(visible=False, elem_classes="panel-box") as panel_settings:
                gr.HTML("""
                <div class="tab-banner" style="background:
                    linear-gradient(170deg,rgba(8,18,50,0.55) 0%,rgba(14,30,80,0.72) 100%),
                    url('https://images.unsplash.com/photo-1518770660439-4636190af475?w=1400&q=85') center/cover no-repeat;">
                  <div class="tab-banner-title">System Configuration</div>
                  <div class="tab-banner-sub">Current runtime configuration of models, chunking parameters and infrastructure paths.</div>
                </div>
                """)
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="section-title">Model Settings</div>')
                        gr.HTML(f"""
                        <div class="config-item"><div class="config-key">Embedding Model</div><div class="config-value">{Config.EMBEDDING_MODEL}</div></div>
                        <div class="config-item"><div class="config-key">LLM Model</div><div class="config-value">{Config.LLAMA_MODEL}</div></div>
                        <div class="config-item"><div class="config-key">LLM Provider</div><div class="config-value">{Config.LLM_PROVIDER.upper()}</div></div>
                        """)
                    with gr.Column():
                        gr.HTML('<div class="section-title">Chunking and Storage</div>')
                        gr.HTML(f"""
                        <div class="config-item"><div class="config-key">Chunk Size</div><div class="config-value">{Config.CHUNK_SIZE} tokens</div></div>
                        <div class="config-item"><div class="config-key">Chunk Overlap</div><div class="config-value">{Config.CHUNK_OVERLAP} tokens</div></div>
                        <div class="config-item"><div class="config-key">FAISS Index Path</div><div class="config-value">{Config.FAISS_INDEX_PATH}</div></div>
                        """)
                    with gr.Column():
                        gr.HTML('<div class="section-title">Marks Distribution</div>')
                        gr.HTML("".join(
                            f'<div class="config-item"><div class="config-key">{part}</div>'
                            f'<div class="config-value">{cfg["questions"]} x {cfg["marks_each"]} marks = {cfg["total"]} marks</div></div>'
                            for part, cfg in Config.MARKS_DISTRIBUTION.items()
                        ))

            # ── FOOTER ───────────────────────────────────────────────────
            gr.HTML("""
            <div style="text-align:center;padding:26px 0 10px;
                        border-top:1px solid rgba(210,180,100,0.18);margin-top:36px;">
              <div style="font-family:'Playfair Display',serif;font-size:20px;color:#e8c96a;margin-bottom:8px;">QuestionHub AI</div>
              <div style="font-size:12px;color:rgba(240,235,220,0.38);letter-spacing:1.5px;text-transform:uppercase;">
                RAG &nbsp;&middot;&nbsp; FAISS &nbsp;&middot;&nbsp; LangGraph &nbsp;&middot;&nbsp; CrewAI &nbsp;&middot;&nbsp; Groq LLM &nbsp;&middot;&nbsp; Gradio
              </div>
            </div>
            """)

            # ── NAV SWITCH LOGIC ──────────────────────────────────────────
            all_panels = [panel_upload, panel_generate, panel_analytics, panel_settings]
            all_btns   = [btn_upload, btn_generate, btn_analytics, btn_settings]

            def switch(idx):
                """Return visibility updates for all panels and active CSS for all cards."""
                panels = [gr.update(visible=(i == idx)) for i in range(4)]
                cards  = [
                    gr.update(elem_classes="nav-card nav-card-active" if i == idx else "nav-card")
                    for i in range(4)
                ]
                return panels + cards

            for i, btn in enumerate(all_btns):
                btn.click(
                    fn=lambda idx=i: switch(idx),
                    inputs=[],
                    outputs=all_panels + all_btns
                )

            # ── FUNCTIONAL EVENT HANDLERS ─────────────────────────────────
            upload_btn.click(
                fn=self.upload_documents,
                inputs=[file_upload],
                outputs=[upload_status]
            )

            def generate_and_prepare(subject, department, exam_type, duration,
                                     total_marks, difficulty, bloom_level, use_crew, use_workflow):
                q_text, analysis, pdf_path = self.generate_questions(
                    subject, department, exam_type, duration,
                    total_marks, difficulty, bloom_level, use_crew, use_workflow
                )
                if pdf_path and os.path.isfile(pdf_path):
                    fname = os.path.basename(pdf_path)
                    status_html = (
                        f'<div style="color:#a8e6a3;font-size:13px;padding:8px 0;font-weight:600;">'
                        f'PDF ready: {fname}</div>'
                    )
                    return q_text, analysis, status_html, gr.update(value=pdf_path, visible=True)
                else:
                    status_html = '<div style="color:#e8a0a0;font-size:13px;padding:8px 0;">PDF generation failed. Please try again.</div>'
                    return q_text, analysis, status_html, gr.update(value=None, visible=False)

            generate_btn.click(
                fn=generate_and_prepare,
                inputs=[subject, department, exam_type, duration, total_marks,
                        difficulty, bloom_level, use_crew, use_workflow],
                outputs=[questions_output, analysis_output, download_btn, pdf_file_out]
            )

            analytics_btn.click(
                fn=self.get_analytics_data,
                outputs=[analytics_text, chart_overview, chart_difficulty, chart_bloom, chart_timeline]
            )

        return demo

def main():
    """Main function to run the application"""
    
    # Initialize application
    app = QuestionPaperGeneratorApp()
    
    # Create Gradio interface
    demo = app.create_gradio_interface()
    
    # Launch application
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()