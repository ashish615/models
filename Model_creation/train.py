"""Training loop for MiniLLM."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import ModelConfig
from model import MiniLLM
from data import get_dataloaders, decode


def evaluate(model, val_loader, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train():
    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, enc, dec = get_dataloaders(cfg)

    model = MiniLLM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
    ).to(device)

    print(f"Parameters: {model.num_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss, total_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:>3} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg.__dict__,
            }, cfg.checkpoint_path)
            print(f"  -> Saved checkpoint (val_loss={val_loss:.4f})")

        # Generate a sample every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            prompt = "the cat"
            prompt_ids = torch.tensor([enc(prompt)], dtype=torch.long, device=device)
            generated = model.generate(prompt_ids, max_new_tokens=50,
                                       temperature=cfg.temperature, top_k=cfg.top_k)
            print(f"  Sample: {repr(dec(generated[0].tolist()))}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
