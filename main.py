"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - COMPLETE BACKEND API
======================================================================
FastAPI backend with PostgreSQL, bcrypt authentication, question scouting,
and AI-judged answer evaluation with 3-attempt progression and self-healing
schema migrations.
======================================================================
"""
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Generator, Optional
import bcrypt
import jwt
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request, status
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

# Force SQLAlchemy to use the installed psycopg2 driver
if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

connect_args = {"check_same_thread": False} if CLEAN_DB_URL.startswith("sqlite") else {}

# pool_pre_ping=True recovers from stale connections caused by cloud database idle timeouts
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
    SQLAlchemy model tracking generated questions per user.
    Why: Matches existing PostgreSQL table schema exactly to avoid UndefinedColumn errors.
    """
    __tablename__ = "question_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    topic = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class CandidateAnswerSubmission(DraftBase):
    """
    SQLAlchemy model storing candidate responses and AI evaluation scores.
    Why: Maintains permanent audit logs of answers and performance for review in DBeaver.
    """
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    username = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    attempt_number = Column(Integer, default=1)
    is_correct = Column(Boolean, default=False)
    feedback = Column(Text, nullable=True)
    ideal_answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Boot-time table creation & Self-Healing Column Migration
try:
    DraftBase.metadata.create_all(bind=engine)
    server_logger.info("Base tables verified.")

    # Automatically adds missing columns if question_answers was created earlier with fewer fields
    with engine.connect() as conn:
        for col_sql in [
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS user_id INTEGER;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS username VARCHAR;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS attempt_number INTEGER DEFAULT 1;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS is_correct BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS feedback TEXT;",
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS ideal_answer TEXT;"
        ]:
            try:
                conn.execute(text(col_sql))
            except Exception:
                pass
        conn.commit()
    server_logger.info("Self-healing auto-migration for question_answers completed.")
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
    Why: Allows seamless support for both logged-in GMs and guest users without throwing 401 on public endpoints.
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
    """Payload for submitting technical answers with attempt count."""
    question: str = Field(..., min_length=3, description="The interview question text")
    answer: str = Field(..., min_length=1, description="The candidate's detailed answer")
    attempt: Optional[int] = Field(default=1, ge=1, le=3, description="Current attempt number (1-3)")
    topic: Optional[str] = None
    difficulty: Optional[str] = None

# 7. FastAPI App & Middleware
app = FastAPI(title="Data Science Draft API - Evaluator", version="3.1.0")
ai_scout_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 8. AI Answer Evaluation Engine
def evaluate_candidate_answer(question: str, candidate_answer: str) -> dict:
    """
    Evaluates candidate response using GPT-4o-mini structured JSON.
    Why: Impartially judges correctness, delivers constructive feedback, and provides canonical answers.
    """
    try:
        system_prompt = (
            "You are an expert Data Science interviewer and judge. Evaluate the candidate's answer to the technical question. "
            "You MUST return a valid JSON object with exactly three keys:\n"
            "- \"is_correct\": boolean (true if candidate answer is conceptually sound, accurate, and covers the core concept; false if incorrect, superficial, or misleading)\n"
            "- \"feedback\": string (1-2 sentences of coaching feedback explaining specifically what was accurate or what key concepts/assumptions were missing)\n"
            "- \"ideal_answer\": string (a clear, canonical 2-3 sentence answer explaining the concept with industry best practices)"
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
        return {
            "is_correct": bool(res_json.get("is_correct", False)),
            "feedback": str(res_json.get("feedback", "Evaluation complete.")),
            "ideal_answer": str(res_json.get("ideal_answer", "Canonical answer unavailable."))
        }
    except Exception as exc:
        server_logger.error("AI Evaluation error: %s", exc)
        return {
            "is_correct": False,
            "feedback": "Evaluation system temporarily unavailable. Review core documentation.",
            "ideal_answer": "Please review standard technical literature for this topic."
        }

# 9. API Endpoints

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(fetch_postgres_session)):
    """Liveness probe verifying API and PostgreSQL responsiveness."""
    try:
        db.execute(DraftBase.metadata.tables["users"].select().limit(1))
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

@app.get("/generate_question")
def generate_question_get(
    topic: str = "General Data Science",
    difficulty: str = "Starter",
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """Generates a question via GET."""
    return execute_question_generation(topic, difficulty, db, active_gm)

@app.post("/generate_question")
def generate_question_post(
    req: DraftPickPayload,
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """Generates a question via POST."""
    return execute_question_generation(req.topic or "General Data Science", req.difficulty or "Starter", db, active_gm)

def execute_question_generation(topic: str, difficulty: str, db: Session, active_gm: Optional[GMFranchise]):
    """Internal helper to generate questions with OpenAI and log to PostgreSQL."""
    try:
        past_picks = []
        if active_gm:
            past_picks = db.query(ProspectScoutingLog).filter(
                ProspectScoutingLog.user_id == active_gm.id,
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

        ai_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        api_response = ai_scout_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=ai_messages,
            max_tokens=200,
            temperature=0.7
        )
        generated_q = api_response.choices[0].message.content.strip()

        history_record = ProspectScoutingLog(
            user_id=active_gm.id if active_gm else None,
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

@app.post("/submit_answer", status_code=status.HTTP_201_CREATED)
def submit_candidate_answer(
    payload: AnswerSubmissionPayload,
    db: Session = Depends(fetch_postgres_session),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Evaluates candidate response via AI judge, applies 3-attempt scoring, and stores in PostgreSQL.
    Why: Tells candidate if answer is correct, tracks attempts up to 3, and reveals ideal answer on completion.
    """
    trimmed_answer = payload.answer.strip()
    trimmed_question = payload.question.strip()

    if not trimmed_answer:
        raise HTTPException(status_code=400, detail="Answer cannot be blank.")
    if not trimmed_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    current_attempt = payload.attempt or 1

    # 1. Evaluate with AI Judge
    eval_result = evaluate_candidate_answer(trimmed_question, trimmed_answer)
    is_correct = eval_result["is_correct"]
    feedback = eval_result["feedback"]
    ideal_answer = eval_result["ideal_answer"]

    # Reveal canonical answer if correct OR on 3rd attempt
    reveal_ideal = is_correct or current_attempt >= 3

    try:
        user_id = active_gm.id if active_gm else None
        username = active_gm.username if active_gm else "Guest"

        submission = CandidateAnswerSubmission(
            user_id=user_id,
            username=username,
            question_text=trimmed_question,
            answer_text=trimmed_answer,
            attempt_number=current_attempt,
            is_correct=is_correct,
            feedback=feedback,
            ideal_answer=ideal_answer if reveal_ideal else None,
            created_at=datetime.now(timezone.utc)
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)

        server_logger.info("Answer evaluated. ID: %s, Correct: %s, Attempt: %s", submission.id, is_correct, current_attempt)
        return {
            "status": "success",
            "submission_id": submission.id,
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
