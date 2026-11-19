from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHECKPOINT = "./output/checkpoint-30"

QUESTIONS = [
    "What is AI?",
    "What is Machine Learning?",
    "What is generative AI?"
]

def generate(model, tokenizer, instruction, max_new_tokens=100):
    prompt = f"Instruction: {instruction}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Base model ---
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
base_model.eval()

print("\n=== BASE MODEL ===")
for q in QUESTIONS:
    print(f"\nQ: {q}")
    print(f"A: {generate(base_model, tokenizer, q)}")

del base_model

# --- Fine-tuned model from checkpoint ---
print("\nLoading fine-tuned model from checkpoint...")
ft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
ft_model = PeftModel.from_pretrained(ft_model, CHECKPOINT)
ft_model.eval()

print("\n=== FINE-TUNED MODEL (checkpoint-3) ===")
for q in QUESTIONS:
    print(f"\nQ: {q}")
    print(f"A: {generate(ft_model, tokenizer, q)}")
