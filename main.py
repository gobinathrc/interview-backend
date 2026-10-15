"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - BACKEND API
======================================================================
FastAPI backend providing AI question generation and PostgreSQL 
persistence for candidate answers.
======================================================================
"""
import os
import logging
from datetime import datetime, timezone
from typing import Generator
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Environment & Logging Initialization
load_dotenv()
logging.basicConfig(level=logging.INFO)
server_logger = logging.getLogger("InterviewBackend")

# 2. Database Connection Configuration
RAW_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./local_draft.db")

# Force SQLAlchemy to use the psycopg2 driver to avoid driver mismatches
if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

connect_args = {"check_same_thread": False} if CLEAN_DB_URL.startswith("sqlite") else {}

# pool_pre_ping=True tests connection liveness and recovers from Render idle disconnects
engine = create_engine(CLEAN_DB_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 3. Relational Database Models
class QuestionAnswer(Base):
    """
    SQLAlchemy model storing user answers submitted for interview questions.
    Why: Retains audit history of candidate responses for review in PostgreSQL.
    """
    __tablename__ = "question_answers"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Boot-time table creation
try:
    Base.metadata.create_all(bind=engine)
    server_logger.info("Database schema synchronized successfully.")
except Exception as boot_err:
    server_logger.error("Database initialization failed: %s", boot_err)

# 4. Dependency Injection for DB Sessions
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

# 5. Pydantic Schemas
class AnswerSubmissionPayload(BaseModel):
    """
    Validation schema for incoming answer submissions.
    Why: Enforces input sanitization and prevents empty string persistence.
    """
    question: str = Field(..., min_length=3, description="The interview question text")
    answer: str = Field(..., min_length=1, description="The user's response to the question")

# 6. FastAPI App & Middleware
app = FastAPI(title="Data Science Draft API", version="2.0.0")
ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7. Endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(db: Session = Depends(get_db)):
    """
    Liveness probe verifying API and database responsiveness.
    Why: Allows Render health probes and operators to monitor system availability.
    """
    try:
        db.execute(Base.metadata.tables["question_answers"].select().limit(1))
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database connectivity failed: {str(exc)}")

@app.get("/generate_question")
def generate_question():
    """
    Generates a challenging data science interview prompt using OpenAI.
    Why: Supplies dynamic, unrepeated evaluation questions for the candidate front office.
    """
    try:
        system_prompt = (
            "You are an expert Data Science interviewer with 60 years of combined industry and academic experience. "
            "You teach clearly and formulate innovative, highly challenging questions that make candidates think deeply "
            "about core principles and modern AI/ML tradeoffs. Return ONLY the interview question text."
        )
        user_prompt = "Generate one innovative and highly challenging data science interview question."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = ai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        question_text = response.choices[0].message.content
        return {"question": question_text}
    except Exception as exc:
        server_logger.error("OpenAI generation failure: %s", exc)
        return {"question": f"Error generating question: {str(exc)}"}

@app.post("/submit_answer", status_code=status.HTTP_201_CREATED)
def submit_answer(payload: AnswerSubmissionPayload, db: Session = Depends(get_db)):
    """
    Saves a candidate's answer into the PostgreSQL database.
    Why: Persists response data permanently for review and grading in DBeaver/Admin dashboards.
    """
    trimmed_answer = payload.answer.strip()
    trimmed_question = payload.question.strip()

    if not trimmed_answer:
        raise HTTPException(status_code=400, detail="Answer cannot be blank.")
    if not trimmed_question:
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    try:
        record = QuestionAnswer(
            question_text=trimmed_question,
            answer_text=trimmed_answer,
            created_at=datetime.now(timezone.utc)
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        server_logger.info("Persisted answer record ID: %s", record.id)
        return {
            "status": "success",
            "message": "Answer saved to PostgreSQL successfully",
            "submission_id": record.id,
            "created_at": record.created_at.isoformat()
        }
    except Exception as db_err:
        db.rollback()
        server_logger.error("Database commit error: %s", db_err)
        raise HTTPException(status_code=500, detail=f"Database persistence failure: {str(db_err)}")
