# Qwen3-0.6B FastAPI Inference Server

This project runs the **Qwen/Qwen3-0.6B** Large Language Model locally using **FastAPI** and **HuggingFace Transformers**.

Qwen3 is a modern thinking-enabled LLM that requires the latest Transformers.  
To ensure compatibility, this project uses **Python 3.11**, NOT Python 3.13.

---

## ⚠️ Important Note
HuggingFace tokenizers, safetensors, and Transformers **do not support Python 3.13 yet**.  
If you try to use Python 3.13, you will get errors like:

**Always use Python 3.11 or 3.12.**

---

# 🚀 Setup Instructions

## 1. Install Python 3.11 (macOS)
```
brew install python@3.11
```

```
/opt/homebrew/bin/python3.11 -m venv venv3.11

source venv3.11/bin/activate
python --version
```

## Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
## If Anything Breaks → Delete and Recreate venv
```
rm -rf venv3.11
/opt/homebrew/bin/python3.11 -m venv venv3.11
source venv3.11/bin/activate
pip install -r requirements.txt
```

## Running the FastAPI Server

## Always run uvicorn using Python module mode (to avoid Anaconda or system Python accidentally loading):
```
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoint
```
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me something interesting about AI."}'
     ```