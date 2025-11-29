from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from model_loader import generate_response
from train_qlora_module import start_training
import threading

app = FastAPI(
    title="Qwen3-0.6B FastAPI",
    description="Local inference API with thinking mode",
    version="1.0.0",
)


class Prompt(BaseModel):
    prompt: str


class TrainPayload(BaseModel):
    data: list


@app.get("/")
def root():
    return {"message": "Qwen3-0.6B Thinking API is running 🚀"}


@app.post("/generate")
def generate_text(data: Prompt):
    result = generate_response(data.prompt)
    return result


@app.post("/train")
def train_model(payload: TrainPayload):
    threading.Thread(target=start_training, args=(payload.data,)).start()
    return {
        "status": "training started",
        "samples": len(payload.data)
    }
