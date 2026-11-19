"""
Send a text prompt to MiniLLM and print the generated response.

Usage:
    # Interactive mode (type prompts at the >>> prompt):
    python query_model.py

    # Single query via CLI:
    python query_model.py --prompt "the cat sat" --max_new_tokens 60

    # Use a saved checkpoint:
    python query_model.py --checkpoint model_checkpoint.pt --prompt "hello world"
"""

import argparse
import os
import sys
import torch

from config import ModelConfig
from model import MiniLLM
from data import encode, decode


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | None, cfg: ModelConfig, device: torch.device) -> MiniLLM:
    model = MiniLLM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=0.0,
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", float("nan"))
        print(f"Loaded checkpoint '{checkpoint_path}'  (epoch={epoch}, val_loss={val_loss:.4f})")
    else:
        if checkpoint_path:
            print(f"Checkpoint '{checkpoint_path}' not found — using randomly initialised weights.")
        else:
            print("No checkpoint given — using randomly initialised weights.")
        print("Tip: run `python train.py` first to get a trained model.\n")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------------

def query(
    model: MiniLLM,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    if not prompt:
        return ""

    # Truncate prompt to fit within context window
    max_prompt_len = model.max_seq_len - max_new_tokens
    prompt_ids = encode(prompt)[-max_prompt_len:]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)

    generated_ids = out[0, len(prompt_ids):].tolist()
    return decode(generated_ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query MiniLLM with a text prompt")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt (omit for interactive mode)")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint.pt",
                        help="Path to model checkpoint (default: model_checkpoint.pt)")
    parser.add_argument("--max_new_tokens", type=int, default=80,
                        help="Number of tokens to generate (default: 80)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (default: 40)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()

    print("=" * 60)
    print("  MiniLLM Query Interface")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Max new toks : {args.max_new_tokens}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Top-k        : {args.top_k}")
    print("=" * 60)

    model = load_model(args.checkpoint, cfg, device)
    print(f"Model parameters: {model.num_parameters():,}\n")

    def run_query(prompt_text: str):
        response = query(model, prompt_text, args.max_new_tokens,
                         args.temperature, args.top_k, device)
        print(f"\nPrompt   : {repr(prompt_text)}")
        print(f"Response : {repr(response)}")
        print(f"Full     : {repr(prompt_text + response)}")
        print("-" * 60)

    # --- Single-shot mode ---
    if args.prompt is not None:
        run_query(args.prompt)
        return

    # --- Interactive mode ---
    print("Interactive mode — type a prompt and press Enter.")
    print("Commands: :quit  :temp <value>  :topk <value>  :tokens <value>\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input == ":quit":
            print("Bye!")
            break
        elif user_input.startswith(":temp "):
            try:
                args.temperature = float(user_input.split()[1])
                print(f"Temperature set to {args.temperature}")
            except ValueError:
                print("Usage: :temp <float>")
            continue
        elif user_input.startswith(":topk "):
            try:
                args.top_k = int(user_input.split()[1])
                print(f"Top-k set to {args.top_k}")
            except ValueError:
                print("Usage: :topk <int>")
            continue
        elif user_input.startswith(":tokens "):
            try:
                args.max_new_tokens = int(user_input.split()[1])
                print(f"Max new tokens set to {args.max_new_tokens}")
            except ValueError:
                print("Usage: :tokens <int>")
            continue

        run_query(user_input)


if __name__ == "__main__":
    main()
