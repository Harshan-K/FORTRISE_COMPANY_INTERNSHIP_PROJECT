# QuestionHub AI — Intelligent Question Paper Generator

An AI-powered system that automatically generates university-level examination papers from uploaded academic documents using RAG, FAISS vector search, Groq LLM, LangGraph, CrewAI, and a Gradio UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio |
| LLM | Groq (llama-3.1-8b-instant) / OpenAI |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Orchestration | LangGraph |
| Multi-Agent | CrewAI |
| Text Splitting | LangChain Text Splitters |
| PDF Generation | ReportLab |
| Database | SQLite |
| Document Parsing | PyPDF2, python-docx |

---

## Project Structure

```
question_paper_generator/
│
├── app.py                      # Main Gradio application entry point
├── config.py                   # Centralised configuration (paths, models, marks)
├── requirements.txt            # All Python dependencies
├── .env.example                # Environment variable template
│
├── agents/
│   └── crew_agents.py          # CrewAI multi-agent system
│
├── rag/
│   └── rag_pipeline.py         # RAG retrieval + LLM generation
│
├── vectorstore/
│   └── faiss_store.py          # FAISS index management
│
├── workflows/
│   └── langgraph_workflow.py   # LangGraph pipeline orchestration
│
├── utils/
│   ├── document_processor.py   # PDF / DOCX / TXT parsing & chunking
│   └── pdf_generator.py        # ReportLab question paper PDF builder
│
├── database/
│   └── db_manager.py           # SQLite: documents, papers, questions, analytics
│
├── uploads/                    # (git-ignored) user-uploaded documents
├── exports/                    # (git-ignored) generated PDF question papers
└── vectorstore/                # (git-ignored) FAISS index files at runtime
```

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/<your-username>/questionhub-ai.git
cd questionhub-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq key at https://console.groq.com/keys

### 5. Run

```bash
python app.py
```

App opens at **http://127.0.0.1:7860**

---

## How to Use

### Step 1 — Document Upload
- Upload your syllabus PDFs, textbooks, lecture notes, or past papers
- Click **Process and Index Documents**
- Documents are chunked, embedded, and stored in FAISS

### Step 2 — Generate Questions
- Enter subject name, department, exam type, duration, total marks
- Select difficulty level and Bloom's taxonomy level
- Choose generation engine (LangGraph workflow recommended)
- Click **Generate Question Paper**
- Download the formatted PDF

### Step 3 — Analytics
- View total documents, questions, and papers generated
- Visualise difficulty distribution, Bloom's taxonomy spread, and paper timeline

---

## Question Paper Format

```
PART A  —  10 × 2  =  20 marks   (short answer)
PART B  —   5 × 5  =  25 marks   (medium answer)
PART C  —   3 × 10 =  30 marks   (long answer)
PART D  —   1 × 15 =  15 marks   (essay / analysis)
─────────────────────────────────
Total                  90 marks
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key (if using openai) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Chunk overlap in characters |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'faiss'` | `pip install faiss-cpu` |
| `No module named 'langchain.text_splitter'` | `pip install langchain-text-splitters` |
| FAISS dimension mismatch error | Delete `vectorstore/faiss_index.index` and `vectorstore/documents.pkl`, re-upload docs |
| PDF download not working | Ensure `exports/` directory exists and is writable |
| Groq API error | Check `GROQ_API_KEY` is set correctly in `.env` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
