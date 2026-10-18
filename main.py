"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - COMPLETE BACKEND API
======================================================================
FastAPI backend with PostgreSQL, bcrypt authentication, question scouting,
AI-judged answer evaluation with 3-attempt progression, 0-100 scoring,
guest session tracking, personalization (Strengths & Weaknesses analytics),
and self-healing schema migrations.
======================================================================
"""
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Generator, Optional, Dict, Any, List
import bcrypt
import jwt
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Environment & Logging Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO)
server_logger = logging.getLogger("InterviewBackend")

# 2. Database Connection Configuration
RAW_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./local_draft.db")

# Force SQLAlchemy to use the installed psycopg2 driver on Render
if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

connect_args = {"check_same_thread": False} if CLEAN_DB_URL.startswith("sqlite") else {}

# pool_pre_ping=True prevents stale connection crashes from cloud DB idle timeouts
engine = create_engine(CLEAN_DB_URL, pool_pre_ping=True, connect_args=connect_args)
DraftSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
DraftBase = declarative_base()

# 3. Relational Database Models
class GMFranchise(DraftBase):
    """
    SQLAlchemy model storing registered user accounts.
    Why: Manages user authentication and binds candidate questions/answers to specific accounts.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

class ProspectScoutingLog(DraftBase):
    """
    SQLAlchemy model tracking generated questions per user and guest.
    Why: Matches existing PostgreSQL table schema and enables guest tracking.
    """
    __tablename__ = "question_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    guest_id = Column(String, index=True, nullable=True)
    topic = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class CandidateAnswerSubmission(DraftBase):
    """
    SQLAlchemy model storing candidate responses, scores, topics, and personalization telemetry.
    Why: Maintains permanent audit logs of answers and performance for review in DBeaver.
    """
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    guest_id = Column(String, index=True, nullable=True)
    username = Column(String, nullable=True)
    topic = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    attempt_number = Column(Integer, default=1)
    is_correct = Column(Boolean, default=False)
    score = Column(Integer, default=0)         # 0-100 evaluation score
    feedback = Column(Text, nullable=True)
    ideal_answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Boot-time table creation & Self-Healing Column Migration
try:
    DraftBase.metadata.create_all(bind=engine)
    server_logger.info("Base tables verified.")

    # Automatically adds missing columns if tables were created earlier with fewer fields
    with engine.connect() as conn:
        migration_statements = [
            "ALTER TABLE question_logs ADD COLUMN IF NOT EXISTS guest_id VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS user_id INTEGER;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS guest_id VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS username VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS topic VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS difficulty VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS attempt_number INTEGER DEFAULT 1;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS is_correct BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS score INTEGER DEFAULT 0;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS feedback TEXT;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS ideal_answer TEXT;"
        ]
        for col_sql in migration_statements:
            try:
                conn.execute(text(col_sql))
            except Exception:
                pass
        conn.commit()
    server_logger.info("Self-healing auto-migration completed.")
except Exception as boot_err:
    server_logger.error("Database schema boot notice: %s", boot_err)

# 4. Security Configuration
JWT_SECRET = os.getenv("SECRET_KEY", "super_secret_draft_key_2026")
JWT_ALGO = "HS256"

# 5. Dependency Injection
def fetch_postgres_session() -> Generator[Session, None, None]:
    """
    Yields an active database session and guarantees closure upon request completion.
    Why: Prevents PostgreSQL connection pool exhaustion by strictly bounding session lifecycles.
    """
    db = DraftSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_optional_current_user(request: Request, db: Session = Depends(fetch_postgres_session)) -> Optional[GMFranchise]:
    """
    Extracts the authenticated user from the Authorization Bearer header if provided.
    Why: Allows seamless support for both logged-in GMs and guest users without throwing 401.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    raw_token = auth_header.split(" ")[1]
    try:
        decoded_payload = jwt.decode(raw_token, JWT_SECRET, algorithms=[JWT_ALGO])
        gm_id = decoded_payload.get("sub")
        if gm_id is not None:
            return db.query(GMFranchise).filter(GMFranchise.id == int(gm_id)).first()
    except Exception as jwt_err:
        server_logger.warning("Token decode notice: %s", jwt_err)
    return None

def verify_front_office_token(request: Request, db: Session = Depends(fetch_postgres_session)) -> GMFranchise:
    """
    Strict authentication gate for protected endpoints.
    Why: Restricts sensitive actions to authenticated users with valid JWT tokens.
    """
    user = get_optional_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    return user

# 6. Pydantic Schemas
class FranchiseAuthPayload(BaseModel):
    """Payload for user registration and authentication."""
    username: str = Field(..., min_length=3, description="Unique GM username")
    password: str = Field(..., min_length=4, description="Raw account password")

class DraftPickPayload(BaseModel):
    """Payload for generating questions with optional filters."""
    topic: Optional[str] = "General Data Science"
    difficulty: Optional[str] = "Starter"

class AnswerSubmissionPayload(BaseModel):
    """Payload for submitting technical answers with attempt count, topic, and difficulty."""
    question: str = Field(..., min_length=3, description="The interview question text")
    answer: str = Field(..., min_length=1, description="The candidate's detailed answer")
    attempt: Optional[int] = Field(default=1, ge=1, le=3, description="Current attempt number (1-3)")
    topic: Optional[str] = "General Data Science"
    difficulty: Optional[str] = "Starter"

# 7. FastAPI App & Middleware
app = FastAPI(
    title="Data Science Draft API - Evaluator & Personalization Engine",
    description="Backend API with PostgreSQL persistence, 0-100 scoring, guest sessions, and personal strengths/weaknesses profiling.",
    version="5.0.0"
)
ai_scout_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 8. AI Answer Evaluation Engine (with 0-100 Score)
def evaluate_candidate_answer(question: str, candidate_answer: str) -> dict:
    """
    Evaluates candidate response using GPT-4o-mini structured JSON with a 0-100 score.
    Why: Provides granular performance grading beyond binary correct/incorrect so candidates
         understand partial credit and where they stand on the depth of their knowledge.
    """
    try:
        system_prompt = (
            "You are an expert Data Science interviewer and impartial judge. "
            "Evaluate the candidate's answer to the technical question.\n\n"
            "You MUST return a valid JSON object with exactly four keys:\n"
            "- \"score\": integer between 0 and 100 representing the quality of the answer. "
            "Use these bands strictly: "
            "90-100 = comprehensive and precise with edge-case awareness; "
            "70-89 = correct fundamentals with minor gaps; "
            "50-69 = partially correct, core concept present but incomplete; "
            "25-49 = incorrect but shows some domain awareness; "
            "0-24 = off-topic or fundamentally wrong.\n"
            "- \"is_correct\": boolean — true only if score >= 70.\n"
            "- \"feedback\": string — 2-3 sentences of specific coaching feedback explaining "
            "what was accurate and what key concept or precision was missing.\n"
            "- \"ideal_answer\": string — a clear, canonical 2-4 sentence model answer using industry best practices."
        )
        user_prompt = f"Question:\n{question}\n\nCandidate Answer:\n{candidate_answer}"

        response = ai_scout_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        res_json = json.loads(response.choices[0].message.content)

        raw_score = res_json.get("score", 0)
        clamped_score = max(0, min(100, int(raw_score)))

        return {
            "score": clamped_score,
            "is_correct": bool(res_json.get("is_correct", clamped_score >= 70)),
            "feedback": str(res_json.get("feedback", "Evaluation complete.")),
            "ideal_answer": str(res_json.get("ideal_answer", "Canonical answer unavailable."))
        }
    except Exception as exc:
        server_logger.error("AI Evaluation error: %s", exc)
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "Evaluation system temporarily unavailable. Review core documentation.",
            "ideal_answer": "Please review standard technical literature for this topic."
        }

# 9. API Endpoints

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(fetch_postgres_session)):
    """Liveness probe verifying API and PostgreSQL responsiveness."""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database connectivity failed: {str(exc)}")

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_franchise(req: FranchiseAuthPayload, db: Session = Depends(fetch_postgres_session)):
    """Registers a new Front Office account with bcrypt password hashing."""
    try:
        existing = db.query(GMFranchise).filter(GMFranchise.username == req.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already drafted by another GM.")

        salt = bcrypt.gensalt()
        safe_hash = bcrypt.hashpw(req.password.encode("utf-8"), salt).decode("utf-8")

        new_franchise = GMFranchise(username=req.username, password_hash=safe_hash)
        db.add(new_franchise)
        db.commit()
        return {"message": "Franchise created successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration DB Error: {str(e)}")

@app.post("/login")
def login_franchise(req: FranchiseAuthPayload, db: Session = Depends(fetch_postgres_session)):
    """Authenticates user credentials and generates a 24-hour JWT token."""
    try:
        gm = db.query(GMFranchise).filter(GMFranchise.username == req.username).first()
        if not gm or not bcrypt.checkpw(req.password.encode("utf-8"), gm.password_hash.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Incorrect username or password.")

        expiry_time = int(time.time()) + 86400
        access_token = jwt.encode({"sub": str(gm.id), "exp": expiry_time}, JWT_SECRET, algorithm=JWT_ALGO)

        return {"access_token": access_token, "username": gm.username, "user_id": gm.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login DB Error: {str(e)}")

# MATCHES FRONTEND URL: /generate_question (Supports both GET and POST)
@app.get("/generate_question")
def generate_question_get(
    topic: str = "General Data Science",
    difficulty: str = "Starter",
    x_guest_id: Optional[str] = Header(None),
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """Generates a question via GET."""
    return execute_question_generation(topic, difficulty, db, active_gm, x_guest_id)

@app.post("/generate_question")
def generate_question_post(
    req: DraftPickPayload,
    x_guest_id: Optional[str] = Header(None),
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """Generates a question via POST."""
    return execute_question_generation(req.topic or "General Data Science", req.difficulty or "Starter", db, active_gm, x_guest_id)

def execute_question_generation(
    topic: str,
    difficulty: str,
    db: Session,
    active_gm: Optional[GMFranchise],
    guest_id: Optional[str] = None
):
    """Internal helper to generate questions with OpenAI and log to PostgreSQL for user or guest."""
    try:
        past_picks = []
        if active_gm:
            past_picks = db.query(ProspectScoutingLog).filter(
                ProspectScoutingLog.user_id == active_gm.id,
                ProspectScoutingLog.topic == topic
            ).all()
        elif guest_id:
            past_picks = db.query(ProspectScoutingLog).filter(
                ProspectScoutingLog.guest_id == guest_id,
                ProspectScoutingLog.topic == topic
            ).all()

        system_prompt = (
            "You are an expert Data Science interviewer with 60 years of combined industry and academic experience. "
            "You ask innovative, highly challenging technical interview questions. "
            "Do NOT include introductions, preambles, or conversational greetings. Output ONLY the exact interview question."
        )

        user_prompt = f"Topic: {topic}\nTarget Difficulty Level: {difficulty}\n\n"
        if past_picks:
            user_prompt += "Do NOT repeat any of these previously asked questions:\n"
            for pick in past_picks[-5:]:
                user_prompt += f"- {pick.question_text[:120]}...\n"
            user_prompt += "\nGenerate a completely NEW and creative question."
        else:
            user_prompt += "Generate a new challenging interview question."

        api_response = ai_scout_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        generated_q = api_response.choices[0].message.content.strip()

        history_record = ProspectScoutingLog(
            user_id=active_gm.id if active_gm else None,
            guest_id=guest_id if not active_gm else None,
            topic=topic,
            difficulty=difficulty,
            question_text=generated_q,
            created_at=datetime.now(timezone.utc)
        )
        db.add(history_record)
        db.commit()
        db.refresh(history_record)

        return {
            "question": generated_q,
            "question_id": history_record.id,
            "topic": topic,
            "difficulty": difficulty
        }
    except Exception as e:
        server_logger.error("Question generation failed: %s", e)
        return {"question": f"SCOUTING SYSTEM ERROR: {str(e)}", "question_id": None}

# MATCHES FRONTEND URL: /submit_answer
@app.post("/submit_answer", status_code=status.HTTP_201_CREATED)
def submit_candidate_answer(
    payload: AnswerSubmissionPayload,
    x_guest_id: Optional[str] = Header(None),
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Evaluates answer via AI judge, applies 0-100 scoring, 3-attempt progression,
    and stores candidate answers in PostgreSQL for both logged-in users and guests.
    """
    trimmed_answer = payload.answer.strip()
    trimmed_question = payload.question.strip()

    if not trimmed_answer:
        raise HTTPException(status_code=400, detail="Answer cannot be blank.")
    if not trimmed_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    current_attempt = payload.attempt or 1

    # 1. Evaluate with AI Judge (returns score + is_correct + feedback + ideal_answer)
    eval_result = evaluate_candidate_answer(trimmed_question, trimmed_answer)
    score = eval_result["score"]
    is_correct = eval_result["is_correct"]
    feedback = eval_result["feedback"]
    ideal_answer = eval_result["ideal_answer"]

    # Reveal ideal answer only when correct OR final attempt exhausted
    reveal_ideal = is_correct or current_attempt >= 3

    try:
        user_id = active_gm.id if active_gm else None
        guest_id = x_guest_id if not active_gm else None
        username = active_gm.username if active_gm else (f"Guest_{x_guest_id[:6]}" if x_guest_id else "Guest")

        submission = CandidateAnswerSubmission(
            user_id=user_id,
            guest_id=guest_id,
            username=username,
            topic=payload.topic or "General Data Science",
            difficulty=payload.difficulty or "Starter",
            question_text=trimmed_question,
            answer_text=trimmed_answer,
            attempt_number=current_attempt,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            ideal_answer=ideal_answer if reveal_ideal else None,
            created_at=datetime.now(timezone.utc)
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)

        server_logger.info(
            "Answer evaluated. ID: %s | Score: %s/100 | Correct: %s | User: %s | Guest: %s",
            submission.id, score, is_correct, username, guest_id
        )
        return {
            "status": "success",
            "submission_id": submission.id,
            "score": score,
            "is_correct": is_correct,
            "feedback": feedback,
            "attempt": current_attempt,
            "attempts_left": max(0, 3 - current_attempt),
            "ideal_answer": ideal_answer if reveal_ideal else None,
            "created_at": submission.created_at.isoformat()
        }
    except Exception as db_err:
        db.rollback()
        server_logger.error("Failed to commit evaluated answer: %s", db_err)
        raise HTTPException(status_code=500, detail=f"Database submission failed: {str(db_err)}")

# PERSONALIZATION: STRENGTHS & WEAKNESSES SCOUTING REPORT
@app.get("/analytics/scouting_report")
def get_scouting_report(
    x_guest_id: Optional[str] = Header(None),
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Computes personalized candidate analytics, highlighting Strengths, Weaknesses,
    topic mastery percentages, and tailored coaching recommendations.
    """
    query = db.query(CandidateAnswerSubmission)
    if active_gm:
        query = query.filter(CandidateAnswerSubmission.user_id == active_gm.id)
    elif x_guest_id:
        query = query.filter(CandidateAnswerSubmission.guest_id == x_guest_id)
    else:
        return {
            "total_answers": 0,
            "overall_avg_score": 0,
            "strengths": [],
            "weaknesses": [],
            "topic_breakdown": {},
            "recommendation": "Start drafting questions to unlock your personalized scouting report!"
        }

    answers = query.order_by(CandidateAnswerSubmission.created_at.desc()).all()

    if not answers:
        return {
            "total_answers": 0,
            "overall_avg_score": 0,
            "strengths": [],
            "weaknesses": [],
            "topic_breakdown": {},
            "recommendation": "No combine data recorded yet. Pick a drill from the Draft Board to begin!"
        }

    topic_stats: Dict[str, Dict[str, Any]] = {}
    total_score = 0

    for a in answers:
        t = a.topic or "General Data Science"
        if t not in topic_stats:
            topic_stats[t] = {"total_score": 0, "count": 0, "correct_count": 0, "highest_score": 0}
        sc = a.score or 0
        topic_stats[t]["total_score"] += sc
        topic_stats[t]["count"] += 1
        if a.is_correct:
            topic_stats[t]["correct_count"] += 1
        if sc > topic_stats[t]["highest_score"]:
            topic_stats[t]["highest_score"] = sc
        total_score += sc

    strengths = []
    weaknesses = []
    breakdown = {}

    for t, stat in topic_stats.items():
        avg = round(stat["total_score"] / stat["count"], 1)
        pass_rate = round((stat["correct_count"] / stat["count"]) * 100, 1)
        breakdown[t] = {
            "avg_score": avg,
            "attempts": stat["count"],
            "pass_rate": pass_rate,
            "highest_score": stat["highest_score"]
        }
        if avg >= 70:
            tier = "🏆 Elite Lock" if avg >= 90 else "🏈 Starter Grade"
            strengths.append({"topic": t, "avg_score": avg, "grade": tier, "attempts": stat["count"]})
        else:
            tier = "⚠️ Practice Squad" if avg >= 50 else "🛑 Cut Day Risk"
            weaknesses.append({"topic": t, "avg_score": avg, "grade": tier, "attempts": stat["count"]})

    strengths.sort(key=lambda x: x["avg_score"], reverse=True)
    weaknesses.sort(key=lambda x: x["avg_score"])

    if weaknesses:
        rec_topics = [w["topic"] for w in weaknesses[:2]]
        rec = f"Focus your combine training on {', '.join(rec_topics)} where your average score is under 70."
    elif strengths:
        rec = "Exceptional performance across the board! You are performing at a First-Round Draft grade."
    else:
        rec = "Keep taking combine reps to establish statistical significance across all categories."

    return {
        "candidate": active_gm.username if active_gm else (f"Guest ({x_guest_id[:6]})" if x_guest_id else "Guest"),
        "total_answers": len(answers),
        "overall_avg_score": round(total_score / len(answers), 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "topic_breakdown": breakdown,
        "recommendation": rec
    }
