from typing import Dict, Any, List

# Try to import CrewAI, make it optional
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("CrewAI not available. Using fallback implementation.")
    
    # Mock classes for fallback
    class Agent:
        def __init__(self, **kwargs):
            self.config = kwargs
    
    class Task:
        def __init__(self, **kwargs):
            self.config = kwargs
    
    class Crew:
        def __init__(self, **kwargs):
            self.config = kwargs
        
        def kickoff(self):
            return "CrewAI mock execution completed"

class QuestionGenerationAgents:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all agents for question generation"""
        
        # Agent 1: Syllabus Analyzer
        self.syllabus_analyzer = Agent(
            role='Syllabus Analyzer',
            goal='Analyze syllabus and identify key units, topics, and learning outcomes',
            backstory="""You are an expert academic curriculum analyst with deep understanding 
                        of educational structures. You excel at breaking down complex syllabi 
                        into manageable learning units and identifying key concepts.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 2: Retriever Agent  
        self.retriever_agent = Agent(
            role='Context Retriever',
            goal='Retrieve relevant academic content from the knowledge base',
            backstory="""You are a specialized information retrieval expert who can quickly 
                        find and extract the most relevant academic content for question 
                        generation from large document collections.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 3: Question Generator Agent
        self.question_generator = Agent(
            role='Question Generator',
            goal='Generate high-quality academic questions based on retrieved content',
            backstory="""You are a seasoned question paper setter with 20+ years of experience 
                        in creating university-level examinations. You understand Bloom's taxonomy 
                        and can create questions of varying difficulty levels.""",
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 4: Paper Formatter Agent
        self.paper_formatter = Agent(
            role='Paper Formatter',
            goal='Format and structure the final question paper professionally',
            backstory="""You are a meticulous document formatter who specializes in creating 
                        professional academic question papers with proper structure, formatting, 
                        and mark distribution.""",
            verbose=True,
            allow_delegation=False
        )
    
    def create_tasks(self, requirements: Dict[str, Any]) -> List[Task]:
        """Create tasks for the crew"""
        
        # Task 1: Analyze Requirements
        analyze_task = Task(
            description=f"""Analyze the subject '{requirements.get('subject')}' and break down 
                           the requirements for generating a {requirements.get('total_marks')} 
                           marks question paper. Identify key topics and learning outcomes.""",
            agent=self.syllabus_analyzer,
            expected_output="Structured analysis of syllabus with key topics and learning outcomes"
        )
        
        # Task 2: Retrieve Content
        retrieve_task = Task(
            description=f"""Based on the syllabus analysis, retrieve relevant academic content 
                           for {requirements.get('subject')} from the knowledge base. Focus on 
                           content suitable for {requirements.get('difficulty', 'Medium')} 
                           difficulty level.""",
            agent=self.retriever_agent,
            expected_output="Retrieved academic content organized by topics",
            context=[analyze_task]
        )
        
        # Task 3: Generate Questions
        generate_task = Task(
            description=f"""Generate questions for a {requirements.get('total_marks')} marks 
                           paper with difficulty level '{requirements.get('difficulty')}' and 
                           Bloom's level '{requirements.get('bloom_level')}'. Follow the standard 
                           university format with Parts A, B, C, and D.""",
            agent=self.question_generator,
            expected_output="Complete set of questions with proper mark distribution",
            context=[analyze_task, retrieve_task]
        )
        
        # Task 4: Format Paper
        format_task = Task(
            description=f"""Format the generated questions into a professional question paper 
                           for {requirements.get('subject')} - {requirements.get('department')}. 
                           Include proper headers, instructions, and mark distribution.""",
            agent=self.paper_formatter,
            expected_output="Professionally formatted question paper ready for export",
            context=[analyze_task, retrieve_task, generate_task]
        )
        
        return [analyze_task, retrieve_task, generate_task, format_task]
    
    def execute_crew(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the crew workflow"""
        
        # Create tasks
        tasks = self.create_tasks(requirements)
        
        # Create and run crew
        crew = Crew(
            agents=[
                self.syllabus_analyzer,
                self.retriever_agent, 
                self.question_generator,
                self.paper_formatter
            ],
            tasks=tasks,
            verbose=True
        )
        
        try:
            # Execute crew workflow
            result = crew.kickoff()
            
            # Generate actual questions using RAG pipeline
            questions = self.rag_pipeline.generate_questions_batch(requirements)
            
            return {
                'success': True,
                'crew_result': str(result),
                'questions': questions,
                'total_questions': len(questions),
                'requirements': requirements
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'questions': [],
                'requirements': requirements
            }
    
    def analyze_syllabus_simple(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Simple syllabus analysis without full crew execution"""
        
        subject = requirements.get('subject', 'Computer Science')
        
        # Simple topic extraction based on common CS subjects
        topics_map = {
            'Computer Science': ['Programming', 'Data Structures', 'Algorithms', 'Database', 'Networks'],
            'Mathematics': ['Calculus', 'Algebra', 'Statistics', 'Geometry', 'Probability'],
            'Physics': ['Mechanics', 'Thermodynamics', 'Electromagnetism', 'Optics', 'Quantum'],
            'Chemistry': ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry', 'Analytical']
        }
        
        topics = topics_map.get(subject, ['Fundamentals', 'Advanced Topics', 'Applications'])
        
        return {
            'subject': subject,
            'topics': topics,
            'learning_outcomes': [f"Understand {topic}" for topic in topics],
            'analysis_complete': True
        }