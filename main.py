from openai import OpenAI
import os
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=False,
    allow_methods = ["*"],
    allow_headers=["*"],
)
@app.get("/generate_question")
def genrate_question():
    system_prompt = "your the an expext in the data science with around 60 years of experience in the field who still study till today always been upto date in the market teacher for data science students and teaches clearly and ask frame quetion in a invative way were it make students to think hard of the concept which in the industry gives them edge in the feild when they get into the market for jobs in the field of data science and AI "
    user_prompt = "generate one innovative and highly challenging data science interview question ."
    messages = [{"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}]
    response = client.chat.completions.create(model="gpt-3.5-turbo",messages=messages)
    responses = response.choices[0].message.content
    return {"question": responses}
