from typing import Dict, Any, List
from pydantic import BaseModel, Field
import json

# Try to import LangGraph, make it optional
try:
    from langgraph.graph import Graph, StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available. Using fallback implementation.")
    
    # Mock classes for fallback
    class StateGraph:
        def __init__(self, state_class):
            self.state_class = state_class
        
        def add_node(self, name, func):
            pass
        
        def add_edge(self, from_node, to_node):
            pass
        
        def set_entry_point(self, node):
            pass
        
        def compile(self):
            return self
        
        def invoke(self, state):
            return state
    
    END = "END"

class QuestionGenerationState(BaseModel):
    """State model for question generation workflow"""
    requirements: Dict[str, Any] = Field(default_factory=dict)
    syllabus_analysis: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: List[Dict[str, Any]] = Field(default_factory=list)
    generated_questions: List[Dict[str, Any]] = Field(default_factory=list)
    difficulty_analysis: Dict[str, Any] = Field(default_factory=dict)
    duplicate_check_results: List[bool] = Field(default_factory=list)
    formatted_paper: Dict[str, Any] = Field(default_factory=dict)
    pdf_path: str = ""
    workflow_complete: bool = False
    error_messages: List[str] = Field(default_factory=list)

class LangGraphWorkflow:
    def __init__(self, rag_pipeline, vector_store):
        self.rag_pipeline = rag_pipeline
        self.vector_store = vector_store
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(QuestionGenerationState)
        
        # Add nodes
        workflow.add_node("document_analyzer", self.document_analyzer_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("difficulty_controller", self.difficulty_controller_node)
        workflow.add_node("question_generator", self.question_generator_node)
        workflow.add_node("duplicate_checker", self.duplicate_checker_node)
        workflow.add_node("formatter", self.formatter_node)
        workflow.add_node("pdf_export", self.pdf_export_node)
        
        # Define workflow edges
        workflow.set_entry_point("document_analyzer")
        
        workflow.add_edge("document_analyzer", "retriever")
        workflow.add_edge("retriever", "difficulty_controller")
        workflow.add_edge("difficulty_controller", "question_generator")
        workflow.add_edge("question_generator", "duplicate_checker")
        workflow.add_edge("duplicate_checker", "formatter")
        workflow.add_edge("formatter", "pdf_export")
        workflow.add_edge("pdf_export", END)
        
        return workflow.compile()
    
    def document_analyzer_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Analyze document requirements and syllabus"""
        try:
            requirements = state.requirements
            
            # Simple analysis based on subject
            subject = requirements.get('subject', 'Computer Science')
            
            analysis = {
                'subject': subject,
                'total_marks': requirements.get('total_marks', 90),
                'difficulty_level': requirements.get('difficulty', 'Medium'),
                'bloom_level': requirements.get('bloom_level', 'Apply'),
                'topics_identified': self._identify_topics(subject),
                'analysis_timestamp': str(state.requirements.get('timestamp', 'now'))
            }
            
            state.syllabus_analysis = analysis
            
        except Exception as e:
            state.error_messages.append(f"Document analysis error: {str(e)}")
        
        return state
    
    def retriever_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Retrieve relevant context from vector store"""
        try:
            subject = state.syllabus_analysis.get('subject', 'Computer Science')
            topics = state.syllabus_analysis.get('topics_identified', ['General'])
            
            retrieved_docs = []
            
            # Retrieve context for each topic
            for topic in topics[:5]:  # Limit to 5 topics
                query = f"{subject} {topic}"
                docs = self.rag_pipeline.retrieve_context(query, k=2)
                retrieved_docs.extend(docs)
            
            state.retrieved_context = retrieved_docs
            
        except Exception as e:
            state.error_messages.append(f"Retrieval error: {str(e)}")
        
        return state
    
    def difficulty_controller_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Control and analyze difficulty distribution"""
        try:
            difficulty = state.requirements.get('difficulty', 'Medium')
            
            # Define difficulty distribution
            difficulty_config = {
                'Easy': {'part_a_ratio': 0.8, 'part_b_ratio': 0.6, 'part_c_ratio': 0.4, 'part_d_ratio': 0.2},
                'Medium': {'part_a_ratio': 0.6, 'part_b_ratio': 0.7, 'part_c_ratio': 0.7, 'part_d_ratio': 0.6},
                'Hard': {'part_a_ratio': 0.4, 'part_b_ratio': 0.5, 'part_c_ratio': 0.8, 'part_d_ratio': 0.9},
                'Mixed': {'part_a_ratio': 0.5, 'part_b_ratio': 0.6, 'part_c_ratio': 0.7, 'part_d_ratio': 0.8}
            }
            
            state.difficulty_analysis = {
                'level': difficulty,
                'distribution': difficulty_config.get(difficulty, difficulty_config['Medium']),
                'bloom_level': state.requirements.get('bloom_level', 'Apply')
            }
            
        except Exception as e:
            state.error_messages.append(f"Difficulty control error: {str(e)}")
        
        return state
    
    def question_generator_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Generate questions based on retrieved context"""
        try:
            questions = self.rag_pipeline.generate_questions_batch(state.requirements)
            state.generated_questions = questions
            
        except Exception as e:
            state.error_messages.append(f"Question generation error: {str(e)}")
            # Provide fallback questions
            state.generated_questions = self._generate_fallback_questions(state.requirements)
        
        return state
    
    def duplicate_checker_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Check for duplicate questions"""
        try:
            duplicate_results = []
            
            for question in state.generated_questions:
                is_duplicate = self.vector_store.check_duplicate_question(
                    question['question'], 
                    threshold=0.8
                )
                duplicate_results.append(is_duplicate)
            
            state.duplicate_check_results = duplicate_results
            
            # Remove duplicates if any found
            filtered_questions = []
            for i, (question, is_duplicate) in enumerate(zip(state.generated_questions, duplicate_results)):
                if not is_duplicate:
                    filtered_questions.append(question)
                else:
                    # Generate alternative question
                    alt_question = self._generate_alternative_question(question, state.requirements)
                    filtered_questions.append(alt_question)
            
            state.generated_questions = filtered_questions
            
        except Exception as e:
            state.error_messages.append(f"Duplicate check error: {str(e)}")
        
        return state
    
    def formatter_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Format questions into proper paper structure"""
        try:
            from config import Config
            
            # Group questions by part
            parts = {'PART_A': [], 'PART_B': [], 'PART_C': [], 'PART_D': []}
            
            for question in state.generated_questions:
                part = question.get('part', 'PART_A')
                if part in parts:
                    parts[part].append(question)
            
            # Format paper
            formatted_paper = {
                'header': {
                    'college_name': 'University Name',
                    'department': state.requirements.get('department', 'Computer Science'),
                    'subject': state.requirements.get('subject', 'Subject Name'),
                    'exam_type': state.requirements.get('exam_type', 'Mid-Term'),
                    'duration': state.requirements.get('duration', '3 Hours'),
                    'total_marks': state.requirements.get('total_marks', 90)
                },
                'instructions': [
                    'Answer all questions.',
                    'Figures to the right indicate full marks.',
                    'Draw diagrams wherever necessary.'
                ],
                'parts': parts,
                'total_questions': len(state.generated_questions)
            }
            
            state.formatted_paper = formatted_paper
            
        except Exception as e:
            state.error_messages.append(f"Formatting error: {str(e)}")
        
        return state
    
    def pdf_export_node(self, state: QuestionGenerationState) -> QuestionGenerationState:
        """Export formatted paper to PDF"""
        try:
            from utils.pdf_generator import PDFGenerator
            import os

            pdf_gen = PDFGenerator()
            pdf_path = pdf_gen.generate_question_paper_pdf(state.formatted_paper)

            # Only set path if the file actually exists
            if pdf_path and os.path.isfile(pdf_path):
                state.pdf_path = pdf_path
            else:
                state.pdf_path = ""

            state.workflow_complete = True
            
        except Exception as e:
            state.error_messages.append(f"PDF export error: {str(e)}")
            state.pdf_path = ""
        
        return state
    
    def _identify_topics(self, subject: str) -> List[str]:
        """Identify topics based on subject"""
        topics_map = {
            'Computer Science': ['Programming', 'Data Structures', 'Algorithms', 'Database', 'Networks'],
            'Mathematics': ['Calculus', 'Algebra', 'Statistics', 'Geometry'],
            'Physics': ['Mechanics', 'Thermodynamics', 'Electromagnetism', 'Optics'],
            'Chemistry': ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry']
        }
        return topics_map.get(subject, ['General Topics'])
    
    def _generate_fallback_questions(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback questions if main generation fails"""
        subject = requirements.get('subject', 'Subject')
        
        fallback_questions = []
        from config import Config
        
        for part, config in Config.MARKS_DISTRIBUTION.items():
            for i in range(config['questions']):
                question = {
                    'question': f"Discuss the fundamental concepts of {subject}. ({config['marks_each']} marks)",
                    'marks': config['marks_each'],
                    'part': part,
                    'difficulty': requirements.get('difficulty', 'Medium'),
                    'bloom_level': requirements.get('bloom_level', 'Apply'),
                    'type': f"{config['marks_each']} mark question"
                }
                fallback_questions.append(question)
        
        return fallback_questions
    
    def _generate_alternative_question(self, original_question: Dict[str, Any], 
                                     requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alternative question to replace duplicate"""
        alt_question = original_question.copy()
        alt_question['question'] = f"Analyze and evaluate the concepts of {requirements.get('subject', 'Subject')}. ({original_question['marks']} marks)"
        return alt_question
    
    def execute_workflow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete workflow"""
        initial_state = QuestionGenerationState(requirements=requirements)

        if LANGGRAPH_AVAILABLE:
            final_state = self.workflow.invoke(initial_state)
        else:
            # Manually run each node in order when LangGraph is not installed
            state = initial_state
            for node_fn in [
                self.document_analyzer_node,
                self.retriever_node,
                self.difficulty_controller_node,
                self.question_generator_node,
                self.duplicate_checker_node,
                self.formatter_node,
                self.pdf_export_node,
            ]:
                state = node_fn(state)
            final_state = state

        return {
            'success': final_state.workflow_complete and len(final_state.error_messages) == 0,
            'questions': final_state.generated_questions,
            'formatted_paper': final_state.formatted_paper,
            'pdf_path': final_state.pdf_path if final_state.pdf_path else None,
            'errors': final_state.error_messages,
            'duplicate_checks': final_state.duplicate_check_results,
            'syllabus_analysis': final_state.syllabus_analysis
        }