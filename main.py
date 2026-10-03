"""
======================================================================
NFL DRAFT: DATA SCIENCE INTERVIEW PLATFORM - BACKEND API
======================================================================
This module powers the FastAPI backend for the Data Science Draft app.
It integrates PostgreSQL for franchise history tracking, OpenAI for 
dynamic question generation, and PyJWT/bcrypt for secure Front Office
authentication.
======================================================================
"""
import os
import time
import bcrypt
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel
from openai import OpenAI

# 1. Environment & DB Initialization
load_dotenv()

# Render PostgreSQL Database URL adjustment
RAW_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./local_draft.db")

# Force SQLAlchemy to explicitly use the psycopg2 driver
if RAW_DB_URL.startswith("postgres://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif RAW_DB_URL.startswith("postgresql://"):
    CLEAN_DB_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
else:
    CLEAN_DB_URL = RAW_DB_URL

draft_db_engine = create_engine(CLEAN_DB_URL)
DraftSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=draft_db_engine)
DraftBase = declarative_base()

# 2. Database Models
class GMFranchise(DraftBase):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

class ProspectScoutingLog(DraftBase):
    __tablename__ = "question_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    topic = Column(String)
    difficulty = Column(String)
    question_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Boot-time table creation with safety net
try:
    DraftBase.metadata.create_all(bind=draft_db_engine)
except Exception as boot_err:
    print(f"CRITICAL BOOT ERROR - DB CONNECTION FAILED: {boot_err}")

# 3. Security Config
JWT_SECRET = os.getenv("SECRET_KEY", "super_secret_draft_key_2026")
JWT_ALGO = "HS256"

# 4. Dependency Injection
def fetch_postgres_session():
    db_session = DraftSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

def verify_front_office_token(request: Request, db: Session = Depends(fetch_postgres_session)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication header.")
    
    raw_token = auth_header.split(" ")[1]
    try:
        decoded_payload = jwt.decode(raw_token, JWT_SECRET, algorithms=[JWT_ALGO])
        gm_id = decoded_payload.get("sub")
        if gm_id is None:
            raise HTTPException(status_code=401, detail="Token payload missing GM ID.")
    except Exception as jwt_err:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(jwt_err)}")
        
    try:
        # Cast to int for precise PostgreSQL querying
        active_gm = db.query(GMFranchise).filter(GMFranchise.id == int(gm_id)).first()
        if not active_gm:
            raise HTTPException(status_code=401, detail="Franchise no longer exists in database.")
        return active_gm
    except Exception as db_err:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(db_err)}")

# 5. FastAPI Application Initialization (Must be before routes!)
app = FastAPI(title="Data Science Draft API")
ai_scout_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Pydantic Schemas
class FranchiseAuthPayload(BaseModel):
    username: str
    password: str

class DraftPickPayload(BaseModel):
    topic: str
    difficulty: str

# 7. API Endpoints
@app.post("/register")
def register_franchise(req: FranchiseAuthPayload, db: Session = Depends(fetch_postgres_session)):
    try:
        existing = db.query(GMFranchise).filter(GMFranchise.username == req.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already drafted by another GM.")
        
        # Secure bcrypt hashing
        salt = bcrypt.gensalt()
        safe_hash = bcrypt.hashpw(req.password.encode('utf-8'), salt).decode('utf-8')
        
        new_franchise = GMFranchise(username=req.username, password_hash=safe_hash)
        db.add(new_franchise)
        db.commit()
        return {"message": "Franchise created successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration DB Error: {str(e)}")

@app.post("/login")
def login_franchise(req: FranchiseAuthPayload, db: Session = Depends(fetch_postgres_session)):
    try:
        gm = db.query(GMFranchise).filter(GMFranchise.username == req.username).first()
        
        if not gm or not bcrypt.checkpw(req.password.encode('utf-8'), gm.password_hash.encode('utf-8')):
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
        
        # Use basic unix timestamps to prevent any datetime timezone bugs
        expiry_time = int(time.time()) + 86400  # 24 hour lifespan
        access_token = jwt.encode({"sub": str(gm.id), "exp": expiry_time}, JWT_SECRET, algorithm=JWT_ALGO)
        
        return {"access_token": access_token, "username": gm.username}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login DB Error: {str(e)}")

@app.post("/generate_question")
def execute_draft_pick(
    req: DraftPickPayload, 
    db: Session = Depends(fetch_postgres_session), 
    active_gm: GMFranchise = Depends(verify_front_office_token)
):
    try:
        past_picks = db.query(ProspectScoutingLog).filter(
            ProspectScoutingLog.user_id == active_gm.id,
            ProspectScoutingLog.topic == req.topic
        ).all()
        
        system_prompt = (
            "You are an expert Data Science interviewer. "
            "You ask innovative, highly challenging questions. "
            "Do NOT include introductions, explanations, or meta-commentary. Output ONLY the exact interview question."
        )
        
        user_prompt = f"Topic: {req.topic}\nTarget Difficulty Level: {req.difficulty}\n\n"
        if past_picks:
            user_prompt += "Do NOT ask any of these previously asked questions:\n"
            for pick in past_picks:
                user_prompt += f"- {pick.question_text[:100]}...\n"
            user_prompt += "\nGenerate a completely NEW question."
        else:
            user_prompt += "Generate a new question."

        ai_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        api_response = ai_scout_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=ai_messages,
            max_tokens=150,
            temperature=0.7
        )
        generated_q = api_response.choices[0].message.content
        
        # Log the question to prevent future duplicates
        history_record = ProspectScoutingLog(
            user_id=active_gm.id,
            topic=req.topic,
            difficulty=req.difficulty,
            question_text=generated_q
        )
        db.add(history_record)
        db.commit()
        
        return {"question": generated_q}
        
    except Exception as e:
        return {"question": f"SCOUTING SYSTEM ERROR: {str(e)}"}
