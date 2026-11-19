# MiniLLM — System Architecture

## Overview

MiniLLM is a GPT-style, decoder-only transformer language model built entirely from PyTorch primitives. It operates at the character level over a vocabulary of 100 printable ASCII characters, trained on synthetically generated text.

---

## Repository Layout

```
Model_creation/
├── config.py            # Hyperparameters and training settings
├── model.py             # Neural network architecture
├── data.py              # Corpus generation, tokenisation, DataLoaders
├── train.py             # Training loop (optimizer, scheduler, checkpointing)
├── query_model.py       # Inference interface (CLI + interactive REPL)
├── test_model.py        # Unit, training, and generation test suite
├── requirements.txt     # Python dependencies
└── model_checkpoint.pt  # Saved model weights (best val_loss)
```

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  Corpus Generators                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │
│  │  Arithmetic  │  │   Simple     │  │   Random     │  │Template│  │
│  │  Sentences   │  │  Sentences   │  │    ASCII     │  │Phrases │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───┬────┘  │
│         └─────────────────┴─────────────────┴──────────────┘       │
│                                    │                                │
│                            generate_corpus()                        │
│                          (6,000 sentences)                          │
│                                    │                                │
│                              encode()                               │
│                     (char → int, VOCAB size=100)                    │
│                                    │                                │
│                           TextDataset                               │
│                     (sliding window, seq_len=64)                    │
│                                    │                                │
│              ┌─────────────────────┴──────────────────┐            │
│              │  random_split (85% / 15%)               │            │
│              ▼                                         ▼            │
│        train_loader                              val_loader         │
│        (batch=32, shuffle=True)             (batch=32, shuffle=F)  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                │
│                                                                     │
│   for each epoch:                                                   │
│     for each batch (x, y):                                         │
│       forward pass → logits, loss                                   │
│       loss.backward()                                               │
│       grad_clip (max_norm=1.0)                                      │
│       optimizer.step()   (AdamW, lr=3e-4, wd=0.01)                 │
│     scheduler.step()     (CosineAnnealingLR)                       │
│     evaluate on val_loader                                          │
│     if val_loss improved → save checkpoint                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE                                     │
│                                                                     │
│   load_model(checkpoint_path)                                       │
│       │                                                             │
│       └─► query(prompt, max_new_tokens, temperature, top_k)        │
│               │                                                     │
│               └─► model.generate()  [autoregressive loop]          │
│                       │                                             │
│                       └─► decode(token_ids) → text output          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture: MiniLLM

```
Input: (B, T) integer token indices
         │
         ▼
┌────────────────────────────────────────────┐
│           Embedding Layer                  │
│                                            │
│  token_embed: Embedding(100, 128)          │
│            +                               │
│  pos_embed:  Embedding(128, 128)           │
│            +                               │
│  embed_drop: Dropout(0.1)                  │
│                                            │
│  Output: (B, T, 128)                       │
└────────────────────┬───────────────────────┘
                     │
         ┌───────────┴───────────┐
         │   × 4 TransformerBlocks│
         │                       │
         │  ┌─────────────────┐  │
         │  │  LayerNorm(128) │  │
         │  └────────┬────────┘  │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │CausalSelfAttn   │  │
         │  │  heads=4        │  │
         │  │  head_dim=32    │  │
         │  └────────┬────────┘  │
         │           │           │
         │      + residual       │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │  LayerNorm(128) │  │
         │  └────────┬────────┘  │
         │           │           │
         │  ┌────────▼────────┐  │
         │  │  FeedForward    │  │
         │  │  128→512→128    │  │
         │  │  GELU + Drop    │  │
         │  └────────┬────────┘  │
         │           │           │
         │      + residual       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   LayerNorm(128)       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   LM Head             │
         │   Linear(128, 100)    │
         │   (weight-tied with   │
         │    token_embed)       │
         └───────────┬───────────┘
                     │
         Output: (B, T, 100) logits
```

---

## CausalSelfAttention Detail

```
Input x: (B, T, 128)
    │
    ├─► QKV Projection: Linear(128, 384, bias=False)
    │       │
    │       └─► split → Q, K, V  each (B, T, 128)
    │               │
    │               └─► reshape to (B, 4, T, 32)  [4 heads, head_dim=32]
    │
    ├─► Attention scores: (Q @ Kᵀ) / √32  →  (B, 4, T, T)
    │
    ├─► Causal mask: upper triangle set to -inf  (prevents future token attention)
    │
    ├─► Softmax → Dropout(0.1)
    │
    ├─► Weighted sum: scores @ V  →  (B, 4, T, 32)
    │
    ├─► Reshape: (B, T, 128)
    │
    └─► Output projection: Linear(128, 128, bias=False) → Dropout(0.1)
```

---

## Module Dependency Graph

```
config.py
    │
    ├──► data.py
    │       └─ generate_corpus, encode, decode, TextDataset, get_dataloaders
    │
    ├──► model.py
    │       └─ CausalSelfAttention
    │          FeedForward
    │          TransformerBlock
    │          MiniLLM
    │
    ├──► train.py
    │       ├─ imports: config, model, data
    │       └─ train(), evaluate()
    │
    ├──► query_model.py
    │       ├─ imports: config, model, data
    │       └─ load_model(), query(), main()
    │
    └──► test_model.py
            ├─ imports: config, model, data
            └─ 15 test functions covering unit, training, generation
```

---

## Checkpoint Format

The file `model_checkpoint.pt` is a PyTorch state dict with the following keys:

| Key                    | Type         | Contents                            |
|------------------------|--------------|-------------------------------------|
| `epoch`                | int          | Epoch at which checkpoint was saved |
| `model_state_dict`     | OrderedDict  | All model weights                   |
| `optimizer_state_dict` | dict         | AdamW optimizer state               |
| `val_loss`             | float        | Validation loss at checkpoint       |
| `config`               | dict         | ModelConfig `__dict__` snapshot     |

---

## Parameter Count (default config)

| Component           | Parameters     |
|---------------------|----------------|
| Token embedding     | 100 × 128 = 12,800 (shared with LM head) |
| Positional embedding| 128 × 128 = 16,384 |
| Per TransformerBlock| ~197,888       |
| 4 × TransformerBlock| ~791,552       |
| Final LayerNorm     | 256            |
| **Total (approx)**  | **~821,000**   |

> Weight tying between `token_embed` and `lm_head` reduces the effective parameter footprint.
