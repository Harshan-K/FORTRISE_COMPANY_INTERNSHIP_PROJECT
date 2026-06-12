from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from config import Config

class PDFGenerator:
    def __init__(self):
        Config.create_directories()
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF generation"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.black
        )
        
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=20,
            alignment=1,
            textColor=colors.darkblue
        )
        
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.black,
            leftIndent=0
        )
        
        self.question_style = ParagraphStyle(
            'QuestionStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.black
        )
        
        self.instruction_style = ParagraphStyle(
            'InstructionStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=0,
            textColor=colors.darkgrey
        )
    
    def create_header_table(self, paper_data: Dict[str, Any]) -> Table:
        """Create header table with college and exam information"""
        header_info = paper_data.get('header', {})
        
        header_data = [
            ['UNIVERSITY NAME', ''],
            ['DEPARTMENT OF ' + header_info.get('department', 'COMPUTER SCIENCE').upper(), ''],
            ['', ''],
            [f"Subject: {header_info.get('subject', 'Subject Name')}", f"Exam: {header_info.get('exam_type', 'Mid-Term')}"],
            [f"Duration: {header_info.get('duration', '3 Hours')}", f"Total Marks: {header_info.get('total_marks', 90)}"],
            [f"Date: {datetime.now().strftime('%d/%m/%Y')}", f"Time: 10:00 AM - 01:00 PM"]
        ]
        
        header_table = Table(header_data, colWidths=[4*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 1), 14),
            ('FONTNAME', (0, 3), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 3), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 3), (-1, -1), 1, colors.black),
        ]))
        
        return header_table
    
    def create_instructions_section(self, instructions: List[str]) -> List:
        """Create instructions section"""
        elements = []
        
        # Instructions heading
        elements.append(Paragraph("INSTRUCTIONS:", self.subheader_style))
        
        # Add instructions
        for i, instruction in enumerate(instructions, 1):
            elements.append(Paragraph(f"{i}. {instruction}", self.instruction_style))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def create_part_section(self, part_name: str, questions: List[Dict[str, Any]], 
                          marks_config: Dict[str, Any]) -> List:
        """Create a section for a specific part"""
        elements = []
        
        if not questions:
            return elements
        
        # Part header
        part_header = f"{part_name} - Answer ALL questions ({marks_config['questions']} × {marks_config['marks_each']} = {marks_config['total']} marks)"
        elements.append(Paragraph(part_header, self.subheader_style))
        elements.append(Spacer(1, 10))
        
        # Questions
        for i, question in enumerate(questions, 1):
            question_text = f"{i}. {question['question']}"
            elements.append(Paragraph(question_text, self.question_style))
            elements.append(Spacer(1, 8))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def generate_question_paper_pdf(self, paper_data: Dict[str, Any]) -> str:
        """Generate complete question paper PDF"""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = paper_data.get('header', {}).get('subject', 'Question_Paper')
        safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        filename = f"Question_Paper_{safe_subject}_{timestamp}.pdf"
        pdf_path = Config.EXPORTS_DIR / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build document elements
        elements = []
        
        # Header
        elements.append(self.create_header_table(paper_data))
        elements.append(Spacer(1, 30))
        
        # Instructions
        instructions = paper_data.get('instructions', [
            'Answer all questions.',
            'Figures to the right indicate full marks.',
            'Draw diagrams wherever necessary.'
        ])
        elements.extend(self.create_instructions_section(instructions))
        
        # Question parts
        parts = paper_data.get('parts', {})
        marks_config = Config.MARKS_DISTRIBUTION
        
        for part_key in ['PART_A', 'PART_B', 'PART_C', 'PART_D']:
            if part_key in parts and parts[part_key]:
                part_elements = self.create_part_section(
                    part_key, 
                    parts[part_key], 
                    marks_config[part_key]
                )
                elements.extend(part_elements)
        
        # Footer information
        elements.append(Spacer(1, 50))
        footer_text = f"Total Questions: {paper_data.get('total_questions', 0)} | " \
                     f"Total Marks: {paper_data.get('header', {}).get('total_marks', 90)}"
        elements.append(Paragraph(footer_text, self.instruction_style))
        
        # Build PDF
        try:
            doc.build(elements)
            return str(pdf_path)
        except Exception as e:
            raise Exception(f"Error generating PDF: {str(e)}")
    
    def generate_simple_pdf(self, questions: List[Dict[str, Any]], 
                          metadata: Dict[str, Any]) -> str:
        """Generate simple PDF without complex formatting"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Simple_Question_Paper_{timestamp}.pdf"
        pdf_path = Config.EXPORTS_DIR / filename
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        elements = []
        
        # Title
        title = f"{metadata.get('subject', 'Question Paper')}"
        elements.append(Paragraph(title, self.title_style))
        elements.append(Spacer(1, 20))
        
        # Questions
        for i, question in enumerate(questions, 1):
            q_text = f"Q{i}. {question['question']} ({question['marks']} marks)"
            elements.append(Paragraph(q_text, self.question_style))
            elements.append(Spacer(1, 10))
        
        doc.build(elements)
        return str(pdf_path)