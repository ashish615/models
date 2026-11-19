import argparse
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

import json

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./output"

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true", help="Ignore existing checkpoint and train from scratch")
args = parser.parse_args()

# Find latest checkpoint
def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

def is_training_complete(checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    if not os.path.isfile(state_file):
        return False
    with open(state_file) as f:
        state = json.load(f)
    return state.get("global_step", 0) >= state.get("max_steps", float("inf"))

checkpoint = None if args.retrain else get_latest_checkpoint(OUTPUT_DIR)

# resume_from is passed to trainer.train() to restore optimizer/scheduler state.
# When a checkpoint is complete we still load its weights but don't restore
# training state, so the Trainer starts fresh steps on top of those weights.
resume_from = checkpoint
if checkpoint:
    if is_training_complete(checkpoint):
        print(f"Checkpoint at {checkpoint} is complete. Continuing training on top of it.")
        resume_from = None
    else:
        print(f"Resuming from checkpoint: {checkpoint}")
else:
    print("Starting fresh training")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto"
)

# LoRA setup: resume adapter weights or init fresh
if checkpoint:
    model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
else:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_config)

# Load dataset
with open("data.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def tokenize(examples):
    text = (
        f"Instruction: {examples['instruction']}\n"
        f"Answer: {examples['output']}"
    )

    result = tokenizer(
        text,
        truncation=True,
        max_length=256
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized = dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=30,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=10,
    learning_rate=2e-4,
    logging_steps=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train(resume_from_checkpoint=resume_from)

model.save_pretrained("./lora_model")