from typing import List, Dict, Any
from vectorstore.faiss_store import VectorStore
from config import Config


def _get_llm_client():
    """Return the configured LLM client (Groq or OpenAI)."""
    provider = Config.LLM_PROVIDER.lower()

    if provider == "groq":
        try:
            from groq import Groq
            if not Config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in your .env file.")
            return ("groq", Groq(api_key=Config.GROQ_API_KEY), Config.GROQ_MODEL)
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

    elif provider == "openai":
        try:
            from openai import OpenAI
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in your .env file.")
            return ("openai", OpenAI(api_key=Config.OPENAI_API_KEY), Config.OPENAI_MODEL)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Use 'groq' or 'openai'.")


def _call_llm(client_tuple, prompt: str) -> str:
    """Send prompt to the LLM and return the text response."""
    provider, client, model = client_tuple
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_client = None
        self._init_llm()

    def _init_llm(self):
        try:
            self.llm_client = _get_llm_client()
            provider = self.llm_client[0]
            model = self.llm_client[2]
            print(f"LLM ready: provider={provider}, model={model}")
        except Exception as e:
            print(f"Warning: LLM not initialized — {e}")
            self.llm_client = None

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector store."""
        return self.vector_store.similarity_search(query, k=k)

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a single context string."""
        if not retrieved_docs:
            return ""
        parts = []
        for i, doc in enumerate(retrieved_docs):
            parts.append(f"[Excerpt {i+1}]\n{doc['content']}")
        return "\n\n".join(parts)

    def generate_question(self, context: str, marks: int, difficulty: str,
                          bloom_level: str, subject: str) -> str:
        """Generate a single question using the real LLM + retrieved context."""

        bloom_descriptions = {
            "Remember":  "factual recall, definitions, listing key terms",
            "Understand": "explanation, interpretation, summarisation",
            "Apply":     "problem-solving, implementation, practical use",
            "Analyze":   "comparison, breakdown, examining relationships",
            "Evaluate":  "critique, assessment, justification",
            "Create":    "design, synthesis, building something new",
        }

        if context:
            context_block = f"""Use ONLY the following content from the uploaded document to frame your question:

{context}

"""
        else:
            context_block = (
                "No specific document content was found for this topic. "
                "Generate a general university-level question on the subject.\n\n"
            )

        prompt = f"""You are an expert university question paper setter for the subject: {subject}.

{context_block}Generate exactly ONE exam question with these specifications:
- Marks: {marks}
- Difficulty: {difficulty}
- Bloom's Taxonomy Level: {bloom_level} ({bloom_descriptions.get(bloom_level, '')})

Rules:
1. The question must be directly based on the document content provided above.
2. Do NOT include the answer, hints, or explanations.
3. Output only the question text — no numbering, no labels, no extra commentary.
4. The question complexity must match a {marks}-mark university exam question."""

        try:
            return _call_llm(self.llm_client, prompt)
        except Exception as e:
            print(f"LLM call failed: {e}")
            return f"Explain the key concepts of {subject} with suitable examples. ({marks} marks)"

    def generate_questions_batch(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all questions for every part based on MARKS_DISTRIBUTION."""
        subject    = requirements.get("subject", "Computer Science")
        difficulty = requirements.get("difficulty", "Medium")
        bloom_level = requirements.get("bloom_level", "Apply")

        questions = []

        for part, cfg in Config.MARKS_DISTRIBUTION.items():
            for _ in range(cfg["questions"]):
                marks = cfg["marks_each"]

                # Retrieve context relevant to this question slot
                query = f"{subject} {bloom_level} {difficulty} {marks} marks"
                docs = self.retrieve_context(query, k=4)
                context = self.format_context(docs)

                question_text = self.generate_question(
                    context=context,
                    marks=marks,
                    difficulty=difficulty,
                    bloom_level=bloom_level,
                    subject=subject,
                )

                questions.append({
                    "question":    question_text,
                    "marks":       marks,
                    "part":        part,
                    "difficulty":  difficulty,
                    "bloom_level": bloom_level,
                    "type":        f"{marks} mark question",
                })

        return questions
