# backend/main.py
"""
FastAPI server for real-time WebSocket interview + /finalize endpoint.
Run:
    uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body

# Attempt to import crew_agent from same folder
from . import crew_agent

app = FastAPI(title="AI Interview WS Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

class InterviewSession:
    def __init__(self, user: Dict[str, Any]):
        self.user = user
        self.questions = []
        self.answers = []
        self.current_q_idx = 0
        self.started = False

    def set_questions(self, qlist):
        self.questions = [{"id": i+1, "text": q} for i,q in enumerate(qlist)]
        self.current_q_idx = 0

    def next_question(self):
        if self.current_q_idx < len(self.questions):
            q = self.questions[self.current_q_idx]
            self.current_q_idx += 1
            return q
        return None

    def record_answer(self, qid: int, text: str):
        self.answers.append({"question_id": qid, "text": text})

# in-memory mapping of websocket -> session (single connection per client)
sessions = {}

@app.websocket("/ws/interview")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = id(ws)
    logger.info("Client connected %s", conn_id)
    session: Optional[InterviewSession] = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type":"agent_message","text":"Invalid JSON"}))
                continue

            mtype = msg.get("type")
            # INIT: start the session & run Crew A to generate questions
            if mtype == "init":
                user = msg.get("user", {})
                session = InterviewSession(user=user)
                sessions[conn_id] = session
                # Generate questions via crew_agent.generate_questions
                # Generate CLEAN questions using new extraction layer
                try:
                    questions = crew_agent.generate_question_list_only(user)
                    if not questions:
                        questions = crew_agent.generate_questions_template(
                            user.get("role", "Unknown"),
                            user.get("interviewLevel", "Medium")
                        )
                except Exception:
                    questions = []

                # If Crew gave nothing → fallback helper
                if not questions:
                    logger.warning("Crew returned no usable questions → using fallback.")
                    questions = crew_agent.fallback_questions(user)

                session.set_questions(questions)
                session.started = True
                logger.info("Session started for user=%s questions=%d", user.get("name"), len(questions))
                q = session.next_question()
                if q:
                    await ws.send_text(json.dumps({"type":"question","question_id":q["id"],"text":q["text"]}))
                else:
                    await ws.send_text(json.dumps({"type":"done","message":"No questions generated"}))

            elif mtype == "answer":
                if not session:
                    await ws.send_text(json.dumps({"type":"agent_message","text":"Session not initialized. Send init first."}))
                    continue
                qid = msg.get("question_id")
                text = msg.get("text","")
                session.record_answer(qid, text)
                logger.info("Recorded answer qid=%s len=%d", qid, len(text))
                # Use interviewer_probe to optionally ask a follow-up
                probe = None
                try:
                    probe = crew_agent.interviewer_probe(session.user, qid, text)
                except Exception:
                    logger.exception("interviewer_probe error")
                if probe:
                    # send probe (same question_id) and do not advance
                    await ws.send_text(json.dumps({"type":"question","question_id":qid,"text":probe,"is_probe":True}))
                else:
                    # send next question
                    q = session.next_question()
                    if q:
                        await ws.send_text(json.dumps({"type":"question","question_id":q["id"],"text":q["text"]}))
                    else:
                        await ws.send_text(json.dumps({"type":"done","message":"All questions completed. Send finalize."}))

            elif mtype == "finalize":
                if not session:
                    await ws.send_text(json.dumps({"type":"agent_message","text":"No active session to finalize."}))
                    continue
                # Build transcript: interleave agent questions and answers
                transcript = []
                # naive mapping: iterate through recorded answers and map to question text
                for ans in session.answers:
                    qid = ans.get("question_id")
                    qtxt = next((q["text"] for q in session.questions if q["id"] == qid), None)
                    if qtxt:
                        transcript.append({"who":"agent","text":qtxt,"ts":""})
                    transcript.append({"who":"user","text":ans.get("text",""),"ts":""})
                # Run post-interview pipeline in background thread to avoid blocking
                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(None, crew_agent.run_interview_pipeline, session.user, transcript)
                    await ws.send_text(json.dumps({"type":"report","report": result.get("final_report")}))
                    await ws.send_text(json.dumps({"type":"agent_message","text":"Report generation complete."}))
                except Exception:
                    logger.exception("run_interview_pipeline failed")
                    # send fallback
                    basic = {"user": session.user, "questions": session.questions, "answers": session.answers, "summary": "fallback report"}
                    await ws.send_text(json.dumps({"type":"report","report":basic}))
            else:
                await ws.send_text(json.dumps({"type":"agent_message","text":"Unknown message type"}))
    except WebSocketDisconnect:
        logger.info("Client disconnected %s", conn_id)
        sessions.pop(conn_id, None)
    except Exception:
        logger.exception("Unhandled error in websocket")
        sessions.pop(conn_id, None)

# Optional HTTP endpoint for finalize (alternative to ws finalize)
@app.post("/finalize")
async def finalize_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Accepts JSON:
    {
      "user": {...},
      "transcript": [ {who:'agent'|'user', text:'', ts:''}, ... ]
    }
    Returns final report object (calls crew_agent.run_interview_pipeline)
    """
    user = payload.get("user")
    transcript = payload.get("transcript", [])
    if not user:
        raise HTTPException(status_code=400, detail="Missing user in payload")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, crew_agent.run_interview_pipeline, user, transcript)
        return {"status":"ok", "result": result}
    except Exception:
        logger.exception("finalize run failed")
        raise HTTPException(status_code=500, detail="Evaluation failed on server")
