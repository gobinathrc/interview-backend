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

# This defines the data React will send to Python
class QuestionRequest(BaseModel):
    topic: str
    difficulty: str
    history: list[str] = []

@app.post("/generate_question")
def generate_question(req: QuestionRequest):
    # Core Persona
    system_prompt = (
        "You are an expert Data Science interviewer with 60 years of experience. "
        "You ask innovative, highly challenging questions that give students a competitive edge. "
        "Do NOT include any introductions, explanations, or meta-commentary. Output ONLY the exact interview question."
    )
    
    # Specific Instructions based on the UI
    user_prompt = f"Topic: {req.topic}\nTarget Difficulty Level: {req.difficulty}\n\n"
    
    # Prevent Repetition
    if req.history:
        user_prompt += "IMPORTANT: Do NOT ask any of these previously asked questions:\n"
        for q in req.history:
            user_prompt += f"- {q[:150]}...\n" # Truncate to save AI memory
        user_prompt += "\nGenerate a completely NEW question based on the topic and difficulty."
    else:
        user_prompt += "Generate a new question based on the topic and difficulty."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    responses = response.choices[0].message.content
    return {"question": responses}
