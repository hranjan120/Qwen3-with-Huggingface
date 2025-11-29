import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


def start_training(json_data):
    import torch
    print("==== QLoRA TRAINING STARTED ====")

    # ------------------------------
    # 1. JSON → HF Dataset
    # ------------------------------
    dataset = Dataset.from_list(json_data)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # Format chat text using Qwen template
    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    dataset = dataset.map(lambda x: {"text": format_chat(x)})

    # ------------------------------
    # 2. Tokenize + Labels (required for Trainer)
    # ------------------------------
    def tokenize_and_add_labels(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding=False
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize_and_add_labels)

    # ------------------------------
    # 3. Force CPU device (QLoRA + MPS = unstable)
    # ------------------------------
    device = "cpu"
    print(f"Using device: {device}")

    # ------------------------------
    # 4. QLoRA BitsAndBytes Config (4-bit)
    # ------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------
    # 5. Load Qwen3 model in 4-bit mode (CPU)
    # ------------------------------
    print("Loading Qwen3 model in 4-bit quantized mode on CPU...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=None,          # VERY IMPORTANT → No MPS / CUDA
    )

    model.to(device)

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # ------------------------------
    # 6. LoRA configuration
    # ------------------------------
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "w_pack",       # fused QKV projection
            "o_proj",       # attention output
            "gate_proj",    # MLP gate
            "up_proj",      # MLP up
            "down_proj"     # MLP down
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------------------------------
    # 7. Disable caching + PyTorch checkpoint fix
    # ------------------------------
    model.config.use_cache = False

    import torch.utils.checkpoint
    torch.utils.checkpoint.use_reentrant = False

    # ------------------------------
    # 8. TrainingArguments (Clean, Mac-safe)
    # ------------------------------
    args = TrainingArguments(
        output_dir="trained-qwen3-qlora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=20,
        num_train_epochs=20,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=200,

        fp16=False,          # CPU → no FP16
        bf16=False,
        use_cpu=True,        # replaces deprecated no_cuda=True
        report_to=[],         # disable wandb/tensorboard
    )

    # ------------------------------
    # 9. Trainer (clean, no warnings)
    # ------------------------------
    print("Starting Trainer...")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        processing_class=tokenizer,  # replaces deprecated tokenizer=
    )

    # ------------------------------
    # 10. TRAIN 🚀
    # ------------------------------
    print("Training started (CPU 4-bit QLoRA)...")
    trainer.train()

    # ------------------------------
    # 11. Save trained LoRA adapters
    # ------------------------------
    model.save_pretrained("trained-qwen3-qlora")
    tokenizer.save_pretrained("trained-qwen3-qlora")

    print("==== QLoRA TRAINING COMPLETED ====")
