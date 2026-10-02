from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import jwt
import os

load_dotenv()

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_draft.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

class QuestionLog(Base):
    __tablename__ = "question_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    topic = Column(String)
    difficulty = Column(String)
    question_text = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# --- SECURITY & AUTH ---
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_draft_key_2026")
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token.")
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user

# --- FASTAPI SETUP ---
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS ---
class AuthRequest(BaseModel):
    username: str
    password: str

class QuestionRequest(BaseModel):
    topic: str
    difficulty: str

# --- ENDPOINTS ---

@app.post("/register")
def register(req: AuthRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == req.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already drafted by another franchise.")
    
    hashed_pw = pwd_context.hash(req.password)
    new_user = User(username=req.username, password_hash=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "Account created successfully!"}

@app.post("/login")
def login(req: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not pwd_context.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    
    # Create JWT Token valid for 24 hours
    token_expires = datetime.utcnow() + timedelta(hours=24)
    token = jwt.encode({"sub": user.id, "exp": token_expires}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "username": user.username}


@app.post("/generate_question")
def generate_question(req: QuestionRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # 1. Fetch user's previous questions directly from the Database!
    past_logs = db.query(QuestionLog).filter(
        QuestionLog.user_id == current_user.id,
        QuestionLog.topic == req.topic
    ).all()
    
    # 2. Build the AI Prompt
    system_prompt = (
        "You are an expert Data Science interviewer. "
        "You ask innovative, highly challenging questions. "
        "Do NOT include introductions, explanations, or meta-commentary. Output ONLY the exact interview question."
    )
    
    user_prompt = f"Topic: {req.topic}\nTarget Difficulty Level: {req.difficulty}\n\n"
    
    if past_logs:
        user_prompt += "Do NOT ask any of these previously asked questions:\n"
        for log in past_logs:
            user_prompt += f"- {log.question_text[:100]}...\n"
        user_prompt += "\nGenerate a completely NEW question."
    else:
        user_prompt += "Generate a new question."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        ai_question = response.choices[0].message.content
        
        # 3. Save the new question to the database under this user's account
        new_log = QuestionLog(
            user_id=current_user.id,
            topic=req.topic,
            difficulty=req.difficulty,
            question_text=ai_question
        )
        db.add(new_log)
        db.commit()
        
        return {"question": ai_question}
        
    except Exception as e:
        return {"question": f"OPENAI ERROR: {str(e)}"}
