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
    # Use default_factory to set timestamp on creation
    timestamp: datetime = PydField(default_factory=datetime.utcnow)

# -------------------------------------------------------
# CALLBACK FUNCTION (NEW)
# -------------------------------------------------------
def log_task_output(task_output):
    """Callback function to log the output of each task to console."""
    try:
        agent_name = task_output.agent.role
    except AttributeError:
        agent_name = "Unknown Agent"
        
    output = task_output.raw
    
    # Log to file
    logger.info(f"[Task Output] Agent: {agent_name}\nOutput:\n{output}")
    
    # Log to console
    print("\n" + "="*50)
    print(f"✅ [Task Output] Agent: {agent_name}")
    print(f"Output:\n{output}")
    print("="*50 + "\n")

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
            logger.info("Trying GROQ qwen/qwen3-32b")
            return LLM(
                model="groq/qwen/qwen3-32b",
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
            "You output a clean list of 8-12 questions. "
            "IMPORTANT: Do NOT include difficulty labels (like 'Medium' or 'Hard') or any other prefix in the question text."
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

    # --- MODIFIED AGENT ---
    report_agent = Agent(
        role="Final Report Compiler",
        backstory="You are a meticulous agent that structures final reports into a Pydantic schema.",
        goal=(
            "Assemble the final structured interview report using the Pydantic FinalReport schema. "
            "Your final answer MUST be ONLY the Pydantic object (as a JSON string) and nothing else. "
            "Do NOT include <think> tags or any other text in your final response."
        ),
        llm=default_llm,
        verbose=True,
    )
    # --- END MODIFICATION ---

    # ---------- Tasks ----------
    task_research = Task(
        description="Research role '{role}' at level '{level}' and identify skills, knowledge areas, and typical interview expectations.",
        agent=researcher_agent,
        expected_output="list"
    )

    task_qgen = Task(
        description="Generate 8-12 tailored interview questions for role '{role}' level '{level}'. "
        "The output must be ONLY the questions, each on a new line, numbered. "
        "Do NOT include difficulty tags, rationales, or any other text.",
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

    # Crew specifically for generating the question list
    crew_qgen = Crew(
        agents=[researcher_agent, qgen_agent],
        tasks=[task_research, task_qgen],
        process=Process.sequential,
        verbose=True,
        task_callback=log_task_output  
    )


    crew_live = Crew(
        agents=[researcher_agent, qgen_agent, interviewer_agent],
        tasks=[task_research, task_qgen, task_interview],
        process=Process.sequential,
        verbose=True,
        task_callback=log_task_output  
    )

    # Post-interview
    task_eval = Task(
        description="Evaluate the provided interview {transcript}. For each 'user' answer, score their knowledge, clarity, tone, and overall performance.",
        agent=evaluator_agent,
        expected_output="dict"
    )

    task_feedback = Task(
        description="Using the evaluated {transcript} from the previous step, provide actionable feedback for each user answer.",
        agent=feedback_agent,
        context=[task_eval],
        expected_output="list"
    )

    # --- MODIFIED TASK ---
    task_report = Task(
        description=(
            "Assemble the final structured interview report for {user} using the evaluated {transcript} and feedback. "
            "Use the Pydantic FinalReport schema. Your output MUST be the raw JSON object, starting with { and ending with }. "
            "Do not add any other text, reasoning, or tags."
        ),
        agent=report_agent,
        context=[task_eval, task_feedback],
        output_pydantic=FinalReport,
        expected_output="FinalReport"
    )
    # --- END MODIFICATION ---


    crew_post = Crew(
        agents=[evaluator_agent, feedback_agent, report_agent],
        tasks=[task_eval, task_feedback, task_report],
        process=Process.sequential,
        verbose=True,
        task_callback=log_task_output 
    )
else:
    crew_live = None
    crew_post = None
    crew_qgen = None # Also set qgen to None


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

# --- MODIFIED FUNCTION ---
def generate_questions(user: Dict[str, Any]) -> List[str]:
    role = user.get("role", "Unknown Role")
    level = user.get("interviewLevel", "Medium")

    logger.info(f"generate_questions: role={role}, level={level}")

    if CREWAI_AVAILABLE and default_llm and crew_qgen: # Check for crew_qgen
        try:
            out = crew_qgen.kickoff(inputs={"role": role, "level": level, "user": user})
            last = out.tasks_output[-1]

            # 1) If crew output is already a list → return it directly
            if hasattr(last, "output") and isinstance(last.output, list):
                logger.info("CrewAI returned a clean list.")
                return last.output

            if hasattr(last, "raw") and isinstance(last.raw, list):
                logger.info("CrewAI returned a raw list.")
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
                
                # --- START OF FIX ---
                # The agent's raw output includes its <think> block.
                # We must strip this block so we only parse the *final* answer.
                think_end = text_block.rfind('</think>')
                if think_end != -1:
                    logger.info("Stripping <think> block from question agent output.")
                    text_block = text_block[think_end + len('</think>'):]
                # --- END OF FIX ---

                extracted = extract_from_text(text_block) # Use helper
                if len(extracted) >= 4:
                    logger.info(f"Extracted {len(extracted)} questions from CrewAI output string.")
                    return extracted
                else:
                    logger.warning(f"Could not extract questions from text block: {text_block}")

        except Exception as e:
            logger.error("CrewAI question generation failed:\n" + traceback.format_exc())

    # Fallback if AI fails or no questions extracted
    logger.warning("Using fallback questions template.")
    return generate_questions_template(role, level)
# --- END MODIFIED FUNCTION ---

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
    if not text: # Add safety check
        return []
        
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
    
    # This should not be reached if generate_questions is working,
    # but as a double-check, we re-run the template.
    logger.warning("generate_questions did not return a list. Using fallback.")
    role = user.get("role", "Unknown Role")
    level = user.get("interviewLevel", "Medium")
    return generate_questions_template(role, level)


def interviewer_probe(user: Dict[str, Any], question_id: int, answer: str) -> Optional[str]:
    try:
        return simple_probe(answer)
    except:
        return None

# --- FULLY REPLACED FUNCTION ---
def run_interview_pipeline(user: Dict[str, Any], transcript: List[Dict[str, Any]]):
    """
    Runs the post-interview evaluation crew and returns the structured report.
    Includes robust parsing logic in case Pydantic output fails.
    """
    logger.info("Running post-interview pipeline…")

    if CREWAI_AVAILABLE and default_llm and crew_post: # Check for crew_post
        try:
            res = crew_post.kickoff(inputs={"user": user, "transcript": transcript})
            last = res.tasks_output[-1]

            # 1. First, try the clean Pydantic output (ideal case)
            if hasattr(last, "pydantic_output") and last.pydantic_output:
                logger.info("CrewAI Pydantic output succeeded.")
                # .model_dump() converts the Pydantic object to a dict
                return {"final_report": last.pydantic_output.model_dump()}

            # 2. If Pydantic output failed, try to parse the agent's raw output string
            logger.warning("Pydantic output failed. Attempting to parse raw agent output.")
            raw_output = last.raw
            
            # Find the JSON block, even if it has <think> tags before it
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = raw_output[json_start:json_end]
                try:
                    # Validate and structure the JSON using our Pydantic model
                    final_report_data = json.loads(json_string)
                    # We must re-validate it with Pydantic to ensure it's correct
                    # This also adds any default values (like timestamp)
                    validated_report = FinalReport(**final_report_data)
                    logger.info("Successfully parsed and validated raw JSON from agent.")
                    return {"final_report": validated_report.model_dump()}
                except Exception as e:
                    logger.error(f"Failed to parse JSON from raw output: {e}\nRaw output was: {raw_output}")
            
            # 3. If both failed, log the raw output and fall through to the fallback
            logger.error(f"Could not find any valid JSON in agent's final output. Raw: {raw_output}")

        except Exception as e:
            logger.error(f"CrewAI evaluator crew failed with exception:\n{traceback.format_exc()}")

    # FALLBACK evaluation (if try block fails or returns nothing)
    logger.warning("CrewAI evaluation failed. Using fallback report generator.")
    questions = []
    answers = []
    qid = 0

    for item in transcript:
        if item["who"] == "agent":
            qid += 1
            questions.append({"id": qid, "text": item["text"]})
        else:
            # Only append answers that have a corresponding question
            if qid > 0:
                answers.append({"question_id": qid, "text": item.get("text", "")})

    # simple evaluation
    scores = {}
    for ans in answers:
        txt = ans.get("text", "") # Use .get for safety
        words = len(txt.split())
        knowledge = min(10, max(1, words / 6))
        clarity = 7.0 if words > 8 else 5.0 # Use floats
        tone = 7.0 # Use float
        overall = round((knowledge + clarity + tone) / 3, 2)
        
        # Match the score structure of EvalScore
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
        # Use Question model for correct structure
        questions=[Question(id=q["id"], text=q["text"], role=None, difficulty=None, tags=[]).model_dump() for q in questions],
        scores={k: EvalScore(**v).model_dump() for k, v in scores.items()},
        summary=f"Fallback evaluation: {len(questions)} questions answered. Avg={avg:.2f}",
        passed=passed,
        timestamp=datetime.utcnow(), # Use correct timestamp
    )

    return {"final_report": final.model_dump()}
# --- END REPLACED FUNCTION ---