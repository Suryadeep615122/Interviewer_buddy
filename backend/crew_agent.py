# backend/crew_agent.py
"""
Gemini + GROQ Hybrid CrewAI Agent System
----------------------------------------

Priority order:
1. Use GOOGLE_API_KEY with Gemini 1.5 Flash (fastest, best)
2. If Gemini fails → fallback to GROQ LLaMA-3.1
3. If both fail → fallback template system (hardcoded questions)

This file fixes:
✓ CrewAI not running 
✓ GROQ being selected incorrectly 
✓ Gemini not loading 
✓ Missing roles & difficulty 
✓ Poor prompts 
✓ Silent failures
"""

import os
import json
import pickle
import traceback
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Load environment variables (important)
from dotenv import load_dotenv
load_dotenv()

# ----- Logging -----
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/crew_agent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crew_agent")

# ----- Try CrewAI -----
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.llm import LLM
    from crewai_tools import SerperDevTool
    CREWAI_AVAILABLE = True
except Exception as e:
    logger.error(f"CrewAI import failed: {e}")
    CREWAI_AVAILABLE = False

# ----- Memory -----
MEMORY_PKL = os.path.join("ai_agents", "memory.pkl")
os.makedirs("ai_agents", exist_ok=True)


def save_memory(key: str, data: Any):
    try:
        mem = {}
        if os.path.exists(MEMORY_PKL):
            with open(MEMORY_PKL, "rb") as f:
                mem = pickle.load(f) or {}
        mem[key] = data
        with open(MEMORY_PKL, "wb") as f:
            pickle.dump(mem, f)
    except Exception:
        logger.error("Failed to save memory\n" + traceback.format_exc())


def load_memory(key: str):
    try:
        if os.path.exists(MEMORY_PKL):
            with open(MEMORY_PKL, "rb") as f:
                mem = pickle.load(f)
                return mem.get(key)
    except:
        pass
    return None


# -------------------------------------------------------
# STRUCTURED OUTPUT MODELS
# -------------------------------------------------------
from pydantic import BaseModel, Field
from pydantic import Field as PydField


class Question(BaseModel):
    id: int
    text: str
    role: Optional[str] = None
    difficulty: Optional[str] = None
    tags: List[str] = PydField(default_factory=list)


class EvalScore(BaseModel):
    knowledge: float = Field(..., ge=0, le=10)
    clarity: float = Field(..., ge=0, le=10)
    tone: float = Field(..., ge=0, le=10)
    overall: float = Field(..., ge=0, le=10)
    comments: Optional[str] = ""


class FinalReport(BaseModel):
    user: Dict[str, Any]
    questions: List[Question]
    scores: Dict[int, EvalScore]
    summary: str
    passed: bool
    timestamp: datetime


# -------------------------------------------------------
# CREWAI SETUP WITH GEMINI + GROQ HYBRID
# -------------------------------------------------------
def make_llm():
    """
    Priority:
    1. Gemini 1.5 Flash
    2. Groq LLaMA 3.1
    """
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # Try Gemini first
    if google_key:
        try:
            logger.info("Trying Gemini 1.5 Flash")
            return LLM(
                model="google/gemini-1.5-flash",
                api_key=google_key,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Failed Gemini LLM: {e}")

    # Try Groq next
    if groq_key:
        try:
            logger.info("Trying GROQ LLaMA-3.1")
            return LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=groq_key,
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Failed Groq LLM: {e}")

    logger.warning("NO LLM WORKED. Will fallback.")
    return None


default_llm = None
tools = []

if CREWAI_AVAILABLE:
    default_llm = make_llm()

    # Serper search tool
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key:
        try:
            tools.append(SerperDevTool(api_key=serper_key))
        except Exception:
            pass

    # ---------- Agents ----------
    researcher_agent = Agent(
        role="Role Research Specialist",
        backstory=(
            "You are an expert job-role researcher. You gather job expectations, "
            "skill requirements, evaluation criteria and real interview patterns "
            "for the given role and difficulty."
        ),
        goal="Deeply research the user's role & level.",
        llm=default_llm,
        tools=tools,
        verbose=True,
    )

    qgen_agent = Agent(
        role="Interview Question Designer",
        backstory=(
            "You design interview questions tailored to the role & difficulty. "
            "You output 8-12 questions with difficulty labels."
        ),
        goal="Generate a structured list of interview questions.",
        llm=default_llm,
        verbose=True,
    )

    interviewer_agent = Agent(
        role="Conversational Interviewer",
        backstory=(
            "Ask questions one-by-one. Read the candidate's answers and optionally "
            "give **one** follow-up (probe). Never repeat the same probe twice."
        ),
        goal="Conduct the interview politely.",
        llm=default_llm,
        verbose=True,
    )

    evaluator_agent = Agent(
        role="Answer Evaluator",
        backstory="Evaluate user answers using rubric scoring.",
        goal="Score responses (knowledge, clarity, tone, overall).",
        llm=default_llm,
        verbose=True,
    )

    feedback_agent = Agent(
        role="Feedback Specialist",
        backstory="You give actionable improvement feedback.",
        goal="Give improvements for each answer.",
        llm=default_llm,
        verbose=True,
    )

    report_agent = Agent(
        role="Final Report Compiler",
        backstory="You structure the final report into Pydantic schema.",
        goal="Build FinalReport object.",
        llm=default_llm,
        verbose=True,
    )

    # ---------- Tasks ----------
    # ---------- Tasks ----------
    task_research = Task(
        description="Research role '{role}' at level '{level}' and identify skills, knowledge areas, and typical interview expectations.",
        agent=researcher_agent,
        expected_output="list"
    )

    task_qgen = Task(
        description="Generate 8-12 tailored interview questions for role '{role}' level '{level}', include difficulty tags and rationale.",
        agent=qgen_agent,
        context=[task_research],
        expected_output="list"
    )

    task_interview = Task(
        description="Simulate interviewer behavior: ask one question at a time, read candidate answers, and produce at most one probe follow-up.",
        agent=interviewer_agent,
        context=[task_qgen],
        expected_output="text"
    )


    crew_live = Crew(
        agents=[researcher_agent, qgen_agent, interviewer_agent],
        tasks=[task_research, task_qgen, task_interview],
        process=Process.sequential,
        verbose=True,
    )

    # Post-interview
    task_eval = Task(
    description="Evaluate each candidate answer and score knowledge, clarity, tone, and overall.",
    agent=evaluator_agent,
    expected_output="dict"
    )

    task_feedback = Task(
        description="Provide actionable feedback for each answer.",
        agent=feedback_agent,
        context=[task_eval],
        expected_output="list"
    )

    task_report = Task(
        description="Assemble the final structured interview report using the Pydantic FinalReport schema.",
        agent=report_agent,
        context=[task_eval, task_feedback],
        output_pydantic=FinalReport,
        expected_output="FinalReport"
    )


    crew_post = Crew(
        agents=[evaluator_agent, feedback_agent, report_agent],
        tasks=[task_eval, task_feedback, task_report],
        process=Process.sequential,
        verbose=True,
    )
else:
    crew_live = None
    crew_post = None


# -------------------------------------------------------
# FALLBACK QUESTIONS
# -------------------------------------------------------
def generate_questions_template(role: str, level: str) -> List[str]:
    base = [
        f"Tell me about yourself and your experience relevant to {role}.",
        f"Why are you interested in this {role} role?",
        f"Describe a recent project where you used important skills for {role}.",
        f"What challenges have you faced in learning or working in {role}?",
    ]

    if level.lower() in ("low", "junior"):
        base += [
            "What are key fundamentals someone must know in this role?",
            "How do you approach debugging or troubleshooting?",
        ]
    elif level.lower() in ("high", "senior"):
        base += [
            "Explain a system you designed and how it scales.",
            "Describe a deep technical decision and your trade-offs.",
        ]
    else:
        base += [
            "How do you optimize performance or reliability?",
            "Explain your testing strategy in detail.",
        ]

    base.append("If asked to elaborate, what more could you add?")
    return base


def simple_probe(answer: str) -> Optional[str]:
    if not answer or len(answer.split()) < 8:
        return "Could you explain with one short example?"
    return None


# -------------------------------------------------------
# MAIN API FUNCTIONS
# -------------------------------------------------------
def generate_questions(user: Dict[str, Any]) -> List[str]:
    role = user.get("role", "Unknown Role")
    level = user.get("interviewLevel", "Medium")

    logger.info(f"generate_questions: role={role}, level={level}")

    if CREWAI_AVAILABLE and default_llm:
        try:
            out = crew_live.kickoff(inputs={"role": role, "level": level, "user": user})
            last = out.tasks_output[-1]

            # 1) If crew output is already a list → return it directly
            if hasattr(last, "output") and isinstance(last.output, list):
                return last.output

            if hasattr(last, "raw") and isinstance(last.raw, list):
                return last.raw

            # 2) Crew output may be a long formatted string containing 1. 2. 3. questions
            text_block = None

            # CrewAI returns different structures depending on model
            if hasattr(last, "output"):
                text_block = last.output
            elif hasattr(last, "raw"):
                text_block = last.raw

            # Convert dict → formatted text if needed
            if isinstance(text_block, dict):
                text_block = json.dumps(text_block, indent=2)

            if isinstance(text_block, str):
                extracted = []
                for line in text_block.split("\n"):
                    line = line.strip()

                    # Matches:
                    # 1. Question
                    # 2) Question
                    # 10. Question text...
                    if len(line) > 2 and line.split(" ")[0].rstrip('.)').isdigit():
                        # remove leading numbering
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            extracted.append(parts[1].strip())

                if len(extracted) >= 4:
                    logger.info("Extracted questions from CrewAI output")
                    return extracted


        except Exception as e:
            logger.error("CrewAI question generation failed:\n" + traceback.format_exc())

    return generate_questions_template(role, level)

# =====================================================================
# NEW: Clean, safe question extractor layer
# =====================================================================

def extract_from_text(text: str):
    """
    Extracts numbered questions from ANY text format:
    - 1. Question
    - 2) Question
    - 10. Question
    - Markdown lists
    - CrewAI JSON dumps
    """
    questions = []
    for line in text.split("\n"):
        stripped = line.strip()

        # Detect patterns like "1. xxx", "1) xxx", "12. xxx"
        prefix = stripped.split(" ")[0].rstrip(".)")
        if prefix.isdigit() and len(stripped) > len(prefix) + 2:
            # Split into number + rest of question
            parts = stripped.split(" ", 1)
            if len(parts) == 2:
                questions.append(parts[1].strip())

    return questions


def generate_question_list_only(user: dict):
    """
    GUARANTEED safe question generator.
    Runs CrewAI → extracts a clean list of questions.
    Falls back automatically if extraction fails.
    """
    raw = generate_questions(user)

    # Already clean list?
    if isinstance(raw, list) and all(isinstance(q, str) for q in raw):
        return raw

    # JSON/dict → dump → extract
    if isinstance(raw, dict):
        try:
            text = json.dumps(raw, indent=2)
            qs = extract_from_text(text)
            if qs:
                return qs
        except:
            pass

    # String → extract
    if isinstance(raw, str):
        qs = extract_from_text(raw)
        if qs:
            return qs

    # If EVERYTHING fails → return empty (backend will fallback)
    return []

def interviewer_probe(user: Dict[str, Any], question_id: int, answer: str) -> Optional[str]:
    try:
        return simple_probe(answer)
    except:
        return None


def run_interview_pipeline(user: Dict[str, Any], transcript: List[Dict[str, Any]]):
    logger.info("Running post-interview pipeline…")

    if CREWAI_AVAILABLE and default_llm:
        try:
            res = crew_post.kickoff(inputs={"user": user, "transcript": transcript})
            last = res.tasks_output[-1]

            if hasattr(last, "pydantic_output") and last.pydantic_output:
                return {"final_report": last.pydantic_output.model_dump()}

        except Exception:
            logger.error("CrewAI evaluator failed:\n" + traceback.format_exc())

    # FALLBACK evaluation
    questions = []
    answers = []
    qid = 0

    for item in transcript:
        if item["who"] == "agent":
            qid += 1
            questions.append({"id": qid, "text": item["text"]})
        else:
            answers.append({"question_id": qid, "text": item["text"]})

    # simple evaluation
    scores = {}
    for ans in answers:
        txt = ans["text"]
        words = len(txt.split())
        knowledge = min(10, max(1, words / 6))
        clarity = 7 if words > 8 else 5
        tone = 7
        overall = round((knowledge + clarity + tone) / 3, 2)
        scores[ans["question_id"]] = {
            "knowledge": round(knowledge, 2),
            "clarity": clarity,
            "tone": tone,
            "overall": overall,
            "comments": "Auto-eval (fallback).",
        }

    avg = sum(v["overall"] for v in scores.values()) / max(len(scores), 1)
    passed = avg >= 6

    final = FinalReport(
        user=user,
        questions=[Question(id=q["id"], text=q["text"]).model_dump() for q in questions],
        scores={k: EvalScore(**v).model_dump() for k, v in scores.items()},
        summary=f"Fallback evaluation: {len(questions)} questions answered. Avg={avg:.2f}",
        passed=passed,
        timestamp=datetime.utcnow(),
    )

    return {"final_report": final.model_dump()}
