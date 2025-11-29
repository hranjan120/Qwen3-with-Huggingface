import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_PATH = "trained-qwen3-qlora"
THINK_END_ID = 151668

print("==> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)


def load_model():
    """Loads fine-tuned LoRA if available, otherwise loads base model."""
    print("==> Checking for fine-tuned LoRA...")

    # Check if directory exists AND contains adapter_model.safetensors
    adapter_exists = (
        os.path.exists(LORA_PATH)
        and os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors"))
    )

    if adapter_exists:
        print("==> LoRA fine-tuned model found! Loading PEFT adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        # Load LoRA
        model = PeftModel.from_pretrained(base_model, LORA_PATH)

        # Optional but recommended → faster inference
        print("==> Merging LoRA into base model...")
        model = model.merge_and_unload()

    else:
        print("==> No LoRA adapters found, loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model


# Load once at startup
model = load_model()


def generate_response(prompt: str):
    messages = [{"role": "user", "content": prompt}]

    # Build chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode generated answer
    output_ids = output[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {"answer": answer}
