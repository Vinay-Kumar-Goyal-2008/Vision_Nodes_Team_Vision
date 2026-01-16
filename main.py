from fastapi import FastAPI
from pydantic import BaseModel

from agents.education_agent import EducationAgent
from ondemand.client import ask_ondemand

app = FastAPI()

class ChatRequest(BaseModel):
    query: str

@app.post("/education")
def education_chat(data: ChatRequest):
    agent = EducationAgent()
    return ask_ondemand(data.query, agent)