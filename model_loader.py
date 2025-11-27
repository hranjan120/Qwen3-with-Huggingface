import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
THINK_END_ID = 151668


print("==> Loading Qwen3-0.6B model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


def generate_response(prompt: str):
    print("==> Generating Response")
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    print("==> Tokenizing input")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Extract newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Separate thinking + final response
    try:
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_ID)
    except ValueError:
        idx = 0

    thinking = tokenizer.decode(
        output_ids[:idx], skip_special_tokens=True).strip()
    answer = tokenizer.decode(
        output_ids[idx:], skip_special_tokens=True).strip()

    return {"thinking": thinking, "answer": answer}
