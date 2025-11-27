from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import generate_response

app = FastAPI(
    title="Qwen3-0.6B FastAPI",
    description="Local inference API with thinking mode",
    version="1.0.0",
)


class Prompt(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {"message": "Qwen3-0.6B Thinking API is running 🚀"}


@app.post("/generate")
def generate_text(data: Prompt):
    result = generate_response(data.prompt)
    return result
