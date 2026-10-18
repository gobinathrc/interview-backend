"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - COMPLETE BACKEND API
======================================================================
FastAPI backend with PostgreSQL, bcrypt authentication, question scouting,
AI-judged answer evaluation with 3-attempt progression, 0-100 scoring,
and self-healing schema migrations.
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

if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

connect_args = {"check_same_thread": False} if CLEAN_DB_URL.startswith("sqlite") else {}

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
    SQLAlchemy model storing candidate responses, AI evaluation scores, and 0-100 scoring.
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
    score = Column(Integer, default=0)         # 0-100 evaluation score
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
            "ALTER TABLE question_answers ADD COLUMN IF NOT EXISTS score INTEGER DEFAULT 0;",
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
    Provides isolated database transaction sessions.
    Why: Ensures session closure and pool reclamation on every request.
    """
    db_session = DraftSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

def verify_front_office_token(
    request: Request,
    db: Session = Depends(fetch_postgres_session)
) -> GMFranchise:
    """
    Validates Bearer token headers and loads the requesting GM record.
    Why: Guards routes against unauthenticated requests.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header."
        )

    raw_token = auth_header.split(" ")[1]
    try:
        decoded_payload = jwt.decode(raw_token, JWT_SECRET, algorithms=[JWT_ALGO])
        gm_id = decoded_payload.get("sub")
        if gm_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload is missing subject ID."
            )
    except Exception as jwt_err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(jwt_err)}"
        )

    try:
        gm_id_int = int(gm_id)
        current_gm = db.query(GMFranchise).filter(GMFranchise.id == gm_id_int).first()
    except (ValueError, TypeError):
        current_gm = db.query(GMFranchise).filter(GMFranchise.username == str(gm_id)).first()

    if not current_gm:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Franchise account no longer exists."
        )

    return current_gm

# 6. Request/Response Schemas
class FrontOfficeCredentials(BaseModel):
    username: str
    password: str

class QuestionDraftRequest(BaseModel):
    topic: str
    difficulty: str

class AnswerSubmissionRequest(BaseModel):
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=1)
    attempt: int = Field(1, ge=1, le=3)

# 7. AI Evaluation Engine
def evaluate_candidate_answer(
    question: str,
    answer: str,
    attempt: int
) -> dict:
    """
    Evaluates a candidate's answer with an integer score (0-100) and actionable feedback.
    Why: Gives a numeric competency score alongside coaching advice and answers.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    eval_prompt = f"""
You are the Head Coach and General Manager at the NFL Data Science Combine.
Grade the candidate's answer to this technical interview question.

Question: {question}
Candidate's Answer: {answer}
Current Attempt: {attempt} of 3

Provide:
1. "score": An integer from 0 to 100 reflecting accuracy, depth, and clarity:
   - 90-100: Elite / First-Round Lock (Complete, highly precise, covers edge cases)
   - 70-89: Starter Grade (Correct core concepts, minor nuances omitted)
   - 50-69: Practice Squad (Partial credit, understands basic premise but missing key steps)
   - 25-49: Training Camp (Weak or superficial, significant gaps)
   - 0-24: Cut Day (Completely incorrect, irrelevant, or hallucinated)
2. "is_correct": boolean (true if score >= 70, false otherwise).
3. "feedback": 2-3 sentences of coaching feedback:
   - If score >= 70: Commend the candidate on specifics they got right.
   - If score < 70 and attempt < 3: Point out what is missing without revealing the full answer.
   - If attempt == 3: Explain what was missing directly.
4. "ideal_answer": A concise 2-4 sentence model solution.

Respond STRICTLY in valid JSON with no markdown wrapping:
{{"score": 85, "is_correct": true, "feedback": "...", "ideal_answer": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(response.choices[0].message.content)
        score = int(parsed.get("score", 0))
        score = max(0, min(100, score))
        is_correct = bool(parsed.get("is_correct", score >= 70))
        return {
            "score": score,
            "is_correct": is_correct,
            "feedback": parsed.get("feedback", "Evaluation completed."),
            "ideal_answer": parsed.get("ideal_answer", "N/A")
        }
    except Exception as err:
        server_logger.warning("OpenAI evaluation fallback: %s", err)
        return {
            "score": 0,
            "is_correct": False,
            "feedback": "Evaluation completed. Please verify your technical keywords and retry.",
            "ideal_answer": "Standard industry practice requires detailed trade-off analysis and algorithmic justification."
        }

# 8. FastAPI Application Setup
app = FastAPI(
    title="NFL Draft: Data Science Combine API",
    description="Backend API with PostgreSQL persistence, 0-100 scoring, and 3-attempt AI evaluation.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 9. API Endpoints
@app.get("/")
def health_check():
    """Returns service health status."""
    return {"status": "active", "service": "NFL Draft Data Science API", "engine": "PostgreSQL"}

@app.post("/register")
def register_franchise(credentials: FrontOfficeCredentials, db: Session = Depends(fetch_postgres_session)):
    """Registers a new user franchise account."""
    existing_user = db.query(GMFranchise).filter(GMFranchise.username == credentials.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Franchise username already claimed.")

    salt = bcrypt.gensalt()
    hashed_pwd = bcrypt.hashpw(credentials.password.encode("utf-8"), salt).decode("utf-8")

    new_gm = GMFranchise(username=credentials.username, password_hash=hashed_pwd)
    db.add(new_gm)
    db.commit()
    db.refresh(new_gm)

    return {"status": "success", "message": f"Franchise '{new_gm.username}' established."}

@app.post("/login")
def login_franchise(credentials: FrontOfficeCredentials, db: Session = Depends(fetch_postgres_session)):
    """Authenticates credentials and issues a JWT token."""
    user = db.query(GMFranchise).filter(GMFranchise.username == credentials.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid franchise username or password.")

    is_valid = bcrypt.checkpw(credentials.password.encode("utf-8"), user.password_hash.encode("utf-8"))
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid franchise username or password.")

    token_payload = {
        "sub": str(user.id),
        "username": user.username,
        "exp": int(time.time()) + (24 * 3600)
    }
    encoded_token = jwt.encode(token_payload, JWT_SECRET, algorithm=JWT_ALGO)

    return {
        "access_token": encoded_token,
        "token_type": "bearer",
        "username": user.username
    }

@app.post("/draft_question")
def draft_question(
    draft_request: QuestionDraftRequest,
    current_gm: GMFranchise = Depends(verify_front_office_token),
    db: Session = Depends(fetch_postgres_session)
):
    """Generates an interview question avoiding duplicates for the current user."""
    prior_logs = db.query(ProspectScoutingLog).filter(
        ProspectScoutingLog.user_id == current_gm.id,
        ProspectScoutingLog.topic == draft_request.topic
    ).all()
    past_questions = [log.question_text for log in prior_logs]

    history_context = ""
    if past_questions:
        history_context = "DO NOT repeat or closely rephrase any of these questions:\n"
        for i, q in enumerate(past_questions[-10:], 1):
            history_context += f"{i}. {q}\n"

    prompt = f"""
You are the Head Scout for an NFL Data Science Front Office.
Generate a high-yield {draft_request.difficulty} interview question on: {draft_request.topic}.
Keep it realistic to real-world machine learning, data engineering, or statistics.
Only output the question text.

{history_context}
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    question_text = response.choices[0].message.content.strip()

    log_entry = ProspectScoutingLog(
        user_id=current_gm.id,
        topic=draft_request.topic,
        difficulty=draft_request.difficulty,
        question_text=question_text
    )
    db.add(log_entry)
    db.commit()

    return {"question": question_text}

@app.post("/submit_answer")
def submit_answer(
    submission: AnswerSubmissionRequest,
    current_gm: GMFranchise = Depends(verify_front_office_token),
    db: Session = Depends(fetch_postgres_session)
):
    """
    Evaluates candidate responses, assigns 0-100 scores, and tracks 3 attempts.
    """
    attempt = submission.attempt
    eval_result = evaluate_candidate_answer(
        question=submission.question,
        answer=submission.answer,
        attempt=attempt
    )

    is_correct = eval_result["is_correct"]
    score = eval_result["score"]
    feedback = eval_result["feedback"]
    ideal_answer = eval_result["ideal_answer"]
    attempts_left = max(0, 3 - attempt)

    # Persist the submission into question_answers table
    try:
        audit_record = CandidateAnswerSubmission(
            user_id=current_gm.id,
            username=current_gm.username,
            question_text=submission.question,
            answer_text=submission.answer,
            attempt_number=attempt,
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            ideal_answer=ideal_answer
        )
        db.add(audit_record)
        db.commit()
    except Exception as persist_err:
        server_logger.error("Failed to persist answer submission: %s", persist_err)
        db.rollback()

    # Mask ideal answer on attempts 1-2 unless candidate passed (score >= 70)
    visible_ideal_answer = ideal_answer if (is_correct or attempt >= 3) else None

    return {
        "status": "success",
        "attempt": attempt,
        "attempts_left": attempts_left,
        "score": score,
        "is_correct": is_correct,
        "feedback": feedback,
        "ideal_answer": visible_ideal_answer,
        "has_attempts_remaining": attempts_left > 0 and not is_correct
    }
