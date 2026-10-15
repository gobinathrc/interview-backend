"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - COMPLETE BACKEND API
======================================================================
Unified FastAPI backend preserving all Front Office GM authentication
(registration, bcrypt password hashing, JWT sessions), question scouting 
history tracking (question_logs), and adding real-time answer persistence 
(question_answers) in PostgreSQL.
======================================================================
"""
import os
import time
import logging
from datetime import datetime, timezone
from typing import Generator, Optional
import bcrypt
import jwt
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Environment & Logging Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO)
server_logger = logging.getLogger("InterviewBackend")

# 2. Database Connection Configuration
RAW_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./local_draft.db")

# Force SQLAlchemy to use the psycopg2 driver to avoid SQLAlchemy 2.x psycopg3 driver mismatch
if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

connect_args = {"check_same_thread": False} if CLEAN_DB_URL.startswith("sqlite") else {}

# pool_pre_ping=True recovers from stale connections caused by cloud database idle timeouts
engine = create_engine(CLEAN_DB_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
    Why: Prevents duplicate questions by comparing against user historical prompts.
    """
    __tablename__ = "question_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    topic = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class CandidateAnswerSubmission(DraftBase):
    """
    SQLAlchemy model storing candidate responses to technical questions.
    Why: Maintains permanent audit logs of answers for review in DBeaver and admin dashboards.
    """
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    username = Column(String, nullable=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Boot-time table creation
try:
    DraftBase.metadata.create_all(bind=engine)
    server_logger.info("All PostgreSQL tables (users, question_logs, question_answers) verified.")
except Exception as boot_err:
    server_logger.error("Database schema boot failed: %s", boot_err)

# 4. Security Configuration
JWT_SECRET = os.getenv("SECRET_KEY", "super_secret_draft_key_2026")
JWT_ALGO = "HS256"

# 5. Dependency Injection
def get_db() -> Generator[Session, None, None]:
    """
    Yields an active database session and guarantees closure upon request completion.
    Why: Prevents PostgreSQL connection pool exhaustion by strictly bounding session lifecycles.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_optional_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[GMFranchise]:
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

def verify_front_office_token(request: Request, db: Session = Depends(get_db)) -> GMFranchise:
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
    """Payload for submitting technical answers."""
    question: str = Field(..., min_length=3, description="The interview question text")
    answer: str = Field(..., min_length=1, description="The candidate's detailed answer")
    topic: Optional[str] = None
    difficulty: Optional[str] = None

# 7. FastAPI App & Middleware
app = FastAPI(title="Data Science Draft API - Full Platform", version="2.5.0")
ai_scout_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 8. API Endpoints

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    """
    Liveness probe verifying API and PostgreSQL responsiveness.
    Why: Allows operators to monitor health and Render to verify web service availability.
    """
    try:
        db.execute(DraftBase.metadata.tables["users"].select().limit(1))
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database connectivity failed: {str(exc)}")

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_franchise(req: FranchiseAuthPayload, db: Session = Depends(get_db)):
    """
    Registers a new Front Office account with bcrypt password hashing.
    Why: Secures user credentials and creates a persistent franchise record in PostgreSQL.
    """
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
def login_franchise(req: FranchiseAuthPayload, db: Session = Depends(get_db)):
    """
    Authenticates user credentials and generates a 24-hour JWT token.
    Why: Grants authenticated access for question history tracking and answer logging.
    """
    try:
        gm = db.query(GMFranchise).filter(GMFranchise.username == req.username).first()
        if not gm or not bcrypt.checkpw(req.password.encode("utf-8"), gm.password_hash.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Incorrect username or password.")

        expiry_time = int(time.time()) + 86400  # 24 hours
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
    db: Session = Depends(get_db),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Generates a technical interview question via GET, checking user history to prevent duplicates.
    Why: Retains backward compatibility for direct GET requests while capturing GM user history.
    """
    return execute_question_generation(topic, difficulty, db, active_gm)

@app.post("/generate_question")
def generate_question_post(
    req: DraftPickPayload,
    db: Session = Depends(get_db),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Generates a technical interview question via POST with customizable topic and difficulty.
    Why: Handles structured frontend payloads and avoids questions previously asked to this GM.
    """
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

        # Log question in question_logs table
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
    db: Session = Depends(get_db),
    active_gm: Optional[GMFranchise] = Depends(get_optional_current_user)
):
    """
    Saves candidate response into the PostgreSQL question_answers table linked to user_id.
    Why: Records evaluations permanently in PostgreSQL for auditing in DBeaver and review dashboards.
    """
    trimmed_answer = payload.answer.strip()
    trimmed_question = payload.question.strip()

    if not trimmed_answer:
        raise HTTPException(status_code=400, detail="Answer cannot be blank.")
    if not trimmed_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    try:
        user_id = active_gm.id if active_gm else None
        username = active_gm.username if active_gm else "Guest"

        submission = CandidateAnswerSubmission(
            user_id=user_id,
            username=username,
            question_text=trimmed_question,
            answer_text=trimmed_answer,
            created_at=datetime.now(timezone.utc)
        )
        db.add(submission)

        if user_id:
            matching_log = db.query(ProspectScoutingLog).filter(
                ProspectScoutingLog.user_id == user_id,
                ProspectScoutingLog.question_text == trimmed_question
            ).first()
            if matching_log:
                matching_log.answer_text = trimmed_answer

        db.commit()
        db.refresh(submission)

        server_logger.info("Answer persisted successfully. Submission ID: %s", submission.id)
        return {
            "status": "success",
            "message": "Answer saved to PostgreSQL successfully",
            "submission_id": submission.id,
            "username": username,
            "user_id": user_id,
            "created_at": submission.created_at.isoformat()
        }
    except Exception as db_err:
        db.rollback()
        server_logger.error("Failed to commit answer submission: %s", db_err)
        raise HTTPException(status_code=500, detail=f"Database submission failed: {str(db_err)}")

@app.get("/my_submissions")
def get_user_submissions(
    db: Session = Depends(get_db),
    active_gm: GMFranchise = Depends(verify_front_office_token)
):
    """
    Fetches all historical submissions for the authenticated GM.
    Why: Empowers candidates to review past answers and evaluate progress over time.
    """
    submissions = db.query(CandidateAnswerSubmission).filter(
        CandidateAnswerSubmission.user_id == active_gm.id
    ).order_by(CandidateAnswerSubmission.created_at.desc()).all()

    return {
        "username": active_gm.username,
        "total_submissions": len(submissions),
        "history": [
            {
                "id": s.id,
                "question": s.question_text,
                "answer": s.answer_text,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in submissions
        ]
    }
