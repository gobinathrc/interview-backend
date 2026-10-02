from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    topic: str
    difficulty: str
    history: list[str] = []

@app.post("/generate_question")
def generate_question(req: QuestionRequest):
    system_prompt = (
        "You are an expert Data Science interviewer. "
        "You ask innovative, highly challenging questions. "
        "Do NOT include introductions, explanations, or meta-commentary. Output ONLY the exact interview question."
    )
    
    user_prompt = f"Topic: {req.topic}\nTarget Difficulty Level: {req.difficulty}\n\n"
    
    if req.history:
        user_prompt += "Do NOT ask any of these previously asked questions:\n"
        for q in req.history:
            user_prompt += f"- {q[:100]}...\n"
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
        return {"question": response.choices[0].message.content}
    except Exception as e:
        # If OpenAI fails, return the error safely as text
        return {"question": f"OPENAI ERROR: {str(e)}"}
