# MiniLLM — Technical Specification

## 1. Project Purpose

MiniLLM is a minimal, self-contained implementation of a GPT-style decoder-only transformer trained to perform next-character prediction (character-level language modelling). The project demonstrates:

- Building a transformer from PyTorch primitives with no external ML libraries
- A complete training pipeline: data generation → tokenisation → training → checkpointing
- Autoregressive text generation with temperature and top-k sampling
- A structured test suite covering architecture correctness, training dynamics, and generation

---

## 2. Environment & Dependencies

| Package     | Minimum Version | Purpose                           |
|-------------|-----------------|-----------------------------------|
| torch       | 2.0.0           | Model, training, tensor ops       |
| numpy       | 1.24.0          | Numerical utilities               |
| matplotlib  | 3.7.0           | (Available for loss plotting)     |
| tqdm        | 4.65.0          | Training progress bar             |
| Python      | 3.10+           | Required for `list[int]` type hints and `str | None` union syntax |

Install:
```bash
pip install -r requirements.txt
```

---

## 3. Configuration (`config.py`)

All hyperparameters are centralised in `ModelConfig`. No CLI argument parsing for config — change values directly.

### 3.1 Vocabulary

| Parameter    | Value | Description                        |
|--------------|-------|------------------------------------|
| `vocab_size` | 100   | `string.printable` — all printable ASCII characters |

### 3.2 Model Architecture

| Parameter    | Value | Description                                     |
|--------------|-------|-------------------------------------------------|
| `embed_dim`  | 128   | Token and position embedding dimension          |
| `num_heads`  | 4     | Attention heads per block; head_dim = 128/4 = 32 |
| `num_layers` | 4     | Number of stacked TransformerBlocks             |
| `ffn_dim`    | 512   | FFN hidden dimension (4× embed_dim)             |
| `max_seq_len`| 128   | Maximum tokens the model can process at once    |
| `dropout`    | 0.1   | Applied in attention, FFN, and embedding        |

### 3.3 Training

| Parameter        | Value   | Description                              |
|------------------|---------|------------------------------------------|
| `batch_size`     | 32      | Samples per gradient step                |
| `learning_rate`  | 3e-4    | Initial AdamW learning rate              |
| `num_epochs`     | 2       | Full passes over the training data       |
| `grad_clip`      | 1.0     | Max gradient norm before clipping        |

### 3.4 Data

| Parameter           | Value | Description                                |
|---------------------|-------|--------------------------------------------|
| `train_seq_len`     | 64    | Context window length for each sample      |
| `num_train_samples` | 2000  | (Informational; actual size is corpus-driven) |
| `num_val_samples`   | 400   | (Informational)                            |

### 3.5 Generation

| Parameter     | Value | Description                                    |
|---------------|-------|------------------------------------------------|
| `temperature` | 0.8   | Softmax temperature for sampling (<1 = sharper)|
| `top_k`       | 40    | Restricts sampling to top-k probability mass   |

### 3.6 Paths

| Parameter         | Value                  |
|-------------------|------------------------|
| `checkpoint_path` | `model_checkpoint.pt`  |

---

## 4. Data Pipeline (`data.py`)

### 4.1 Vocabulary & Tokenisation

```python
VOCAB = string.printable          # 100 printable ASCII characters
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB)}
```

- **encode(text)**: maps each character to its index; unknown characters fall back to the space index.
- **decode(tokens)**: maps indices back to characters; unknown indices produce empty string.

### 4.2 Corpus Generation

`generate_corpus(num_sentences=5000)` samples from four generators with equal probability:

| Generator              | Function               | Example output                          |
|------------------------|------------------------|-----------------------------------------|
| Arithmetic expressions | `_arithmetic_sentence` | `"37 + 58 = 95"`, `"12 * 9 = 108"`     |
| Simple sentences       | `_simple_sentence`     | `"the fast dog runs"`                   |
| Random ASCII           | `_random_sentence`     | `"xqpbz amkln oqriv ..."`              |
| Template phrases       | `_template_sentence`   | `"the quick brown fox jumps over the lazy dog"` |

Training runs use `num_sentences=6000`.

### 4.3 Dataset (`TextDataset`)

A sliding-window `torch.utils.data.Dataset`:

- Each sample `(x, y)` covers indices `[i : i+seq_len]` and `[i+1 : i+seq_len+1]` respectively.
- This implements teacher-forcing: model is trained to predict the next character at every position simultaneously.
- `__len__` = `max(0, total_tokens - seq_len)`

### 4.4 DataLoaders

`get_dataloaders(cfg)` returns `(train_loader, val_loader, encode, decode)`.

- 85 / 15 train/val split via `random_split`
- `train_loader`: shuffled, `drop_last=True` (ensures uniform batch sizes)
- `val_loader`: unshuffled, `drop_last=False` (evaluates every sample)

---

## 5. Model Architecture (`model.py`)

### 5.1 `CausalSelfAttention`

**Constructor parameters**: `embed_dim`, `num_heads`, `max_seq_len`, `dropout`

**Key implementation details**:

- Fused QKV linear projection `(embed_dim → 3 × embed_dim)` for efficiency.
- Split output into Q, K, V tensors and reshape to `(B, heads, T, head_dim)`.
- Scaled dot-product: `attn = (Q @ Kᵀ) / √head_dim`
- Causal mask registered as a buffer (non-parameter tensor), shape `(1, 1, max_seq_len, max_seq_len)`, lower-triangular ones. Upper triangle filled with `-inf` before softmax.
- `attn_drop`: applied to attention weights after softmax.
- `resid_drop`: applied to the output projection result.

**Shape flow**:
```
x: (B, T, C)
→ QKV: (B, T, 3C) → split → Q,K,V: (B, T, C)
→ reshape: (B, heads, T, head_dim)
→ attn: (B, heads, T, T) → masked softmax → dropout
→ out: (B, heads, T, head_dim) → (B, T, C)
→ out_proj: (B, T, C)
```

### 5.2 `FeedForward`

Position-wise two-layer MLP:

```
Linear(embed_dim → ffn_dim) → GELU → Linear(ffn_dim → embed_dim) → Dropout
```

GELU is used (rather than ReLU) for smoother gradients and better empirical performance in language models.

### 5.3 `TransformerBlock`

Implements the **Pre-LayerNorm** (Pre-LN) variant:

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-LN places normalisation before the sublayer rather than after, which stabilises training gradients and reduces the need for learning-rate warmup.

### 5.4 `MiniLLM`

**Constructor parameters**: `vocab_size`, `embed_dim`, `num_heads`, `num_layers`, `ffn_dim`, `max_seq_len`, `dropout`

**Layers**:
- `token_embed`: `Embedding(vocab_size, embed_dim)`
- `pos_embed`: `Embedding(max_seq_len, embed_dim)` — learned, not sinusoidal
- `embed_drop`: `Dropout(dropout)`
- `blocks`: `ModuleList` of `num_layers` TransformerBlocks
- `ln_final`: `LayerNorm(embed_dim)`
- `lm_head`: `Linear(embed_dim, vocab_size, bias=False)`

**Weight tying**: `lm_head.weight` is set to `token_embed.weight`. This shares parameters between input and output token representations, reducing total parameter count and is standard practice in language models.

**Weight initialisation**:
- Linear weights: `N(0, 0.02)`
- Embedding weights: `N(0, 0.02)`
- LayerNorm weights: 1.0, biases: 0.0

**Forward pass**:
```python
positions = arange(T).unsqueeze(0)          # (1, T)
x = embed_drop(token_embed(idx) + pos_embed(positions))
for block in blocks:
    x = block(x)
x = ln_final(x)
logits = lm_head(x)                         # (B, T, vocab_size)
loss = cross_entropy(logits, targets)       # if targets provided
```

**Generation** (`model.generate`):

Autoregressive loop producing `max_new_tokens` tokens:
1. Crop context to `max_seq_len` (sliding window).
2. Forward pass, take logits for the last position only.
3. Divide logits by `temperature`.
4. If `top_k` set: zero out all logits below the k-th largest.
5. Apply softmax → multinomial sample → append to sequence.

---

## 6. Training (`train.py`)

### 6.1 Optimizer

**AdamW** with:
- `lr = 3e-4`
- `weight_decay = 0.01`

AdamW decouples weight decay from the gradient update, which is more principled than L2 regularisation in Adam.

### 6.2 Learning Rate Schedule

**CosineAnnealingLR** with `T_max = num_epochs`:

- Starts at `learning_rate` and decays smoothly to near zero following a cosine curve.
- Encourages convergence to a flatter minimum near the end of training.

### 6.3 Training Loop

```
for epoch in 1..num_epochs:
    model.train()
    for (x, y) in train_loader:
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val_loss:
        save checkpoint
    if epoch % 5 == 0:
        generate sample text
```

### 6.4 Gradient Clipping

`torch.nn.utils.clip_grad_norm_` clips the global L2 norm of all gradients to `grad_clip=1.0`. This prevents gradient explosions that can occur during early training or with long sequences.

### 6.5 Checkpointing

The checkpoint is saved whenever `val_loss` improves (not on a fixed schedule). Checkpoint contains:
- `model_state_dict`
- `optimizer_state_dict` (allows training resumption)
- `val_loss`
- `epoch`
- `config` (full `ModelConfig.__dict__` for reproducibility)

---

## 7. Inference (`query_model.py`)

### 7.1 Model Loading

`load_model(checkpoint_path, cfg, device)`:
- Instantiates `MiniLLM` with `dropout=0.0` (inference mode, no stochasticity in dropout).
- Loads `model_state_dict` from checkpoint via `torch.load(..., weights_only=True)`.
- Falls back gracefully to random weights if no checkpoint exists.

### 7.2 Query Function

`query(model, prompt, max_new_tokens, temperature, top_k, device)`:
- Truncates prompt to `max_seq_len - max_new_tokens` characters from the right to guarantee room for generation.
- Encodes prompt → tensor → calls `model.generate()`.
- Returns only the generated tokens (not the prompt echo).

### 7.3 CLI Interface

```bash
# Single query
python query_model.py --prompt "the cat sat" --max_new_tokens 60

# Use specific checkpoint
python query_model.py --checkpoint model_checkpoint.pt --prompt "hello"

# Interactive REPL
python query_model.py
```

**Interactive REPL commands**:

| Command          | Effect                            |
|------------------|-----------------------------------|
| `:quit`          | Exit                              |
| `:temp <float>`  | Set sampling temperature          |
| `:topk <int>`    | Set top-k value                   |
| `:tokens <int>`  | Set max_new_tokens                |

---

## 8. Test Suite (`test_model.py`)

15 tests across three categories, executed sequentially via a custom runner (no pytest required).

### 8.1 Unit Tests

| Test                           | What is verified                                                          |
|--------------------------------|---------------------------------------------------------------------------|
| `test_output_shape`            | `logits` shape is `(B, T, vocab_size)`; loss is positive when targets given |
| `test_no_target_no_loss`       | `loss is None` when targets not passed                                    |
| `test_parameter_count`         | Model has > 0 trainable parameters                                        |
| `test_weight_tying`            | `token_embed.weight` and `lm_head.weight` share the same memory pointer   |
| `test_causal_mask`             | Changing token at position T does not affect logits at positions < T      |
| `test_sequence_length_boundary`| Model processes a full-length sequence of `max_seq_len` without error     |
| `test_exceeds_max_seq_len_raises` | `AssertionError` raised when input exceeds `max_seq_len`              |

### 8.2 Training Tests

| Test                   | What is verified                                                 |
|------------------------|------------------------------------------------------------------|
| `test_loss_decreases`  | Loss drops after 50 gradient steps on a fixed batch             |
| `test_gradient_flow`   | Every named parameter receives a gradient after `loss.backward()`|

### 8.3 Generation Tests

| Test                              | What is verified                                                |
|-----------------------------------|-----------------------------------------------------------------|
| `test_generate_shape`             | Output shape is `(B, prompt_len + max_new_tokens)`             |
| `test_generate_in_vocab`          | All generated token indices are within `[0, vocab_size)`       |
| `test_generate_temperature`       | Low temperature produces fewer unique sequences than high temp  |
| `test_generate_top_k`             | Generation with `top_k=5` produces correct output shape        |
| `test_perplexity_on_corpus`       | Perplexity is finite and < `vocab_size × 4`                   |
| `test_text_generation_from_prompt`| End-to-end: decoded output starts with prompt text             |

**Run all tests**:
```bash
python test_model.py
```

---

## 9. Running the Project

### 9.1 Training

```bash
python train.py
```

Expected console output (per epoch):
```
Device: cpu
Generating dummy text corpus …
  Corpus length : ~240,000 characters
  Token IDs     : ~240,000 tokens
  Dataset size  : ~239,936 samples (seq_len=64)
  Train batches : ~6,373, Val batches: ~1,125
Parameters: ~821,000
Epoch   1 | train_loss=3.1842 | val_loss=3.0641 | lr=0.000300
  -> Saved checkpoint (val_loss=3.0641)
Epoch   2 | train_loss=2.9103 | val_loss=2.8891 | lr=0.000150
  -> Saved checkpoint (val_loss=2.8891)
Training complete. Best val_loss=2.8891
```

### 9.2 Querying

```bash
python query_model.py --prompt "the quick brown" --max_new_tokens 50
```

### 9.3 Testing

```bash
python test_model.py
```

---

## 10. Design Decisions & Trade-offs

| Decision                        | Rationale                                                                 |
|---------------------------------|---------------------------------------------------------------------------|
| Character-level tokenisation    | Simplest possible vocabulary; no tokenizer library needed                 |
| Synthetic corpus                | Enables fully self-contained training without external datasets           |
| Pre-LayerNorm blocks            | More stable gradient flow than Post-LN; standard in modern LLMs           |
| Fused QKV projection            | Single matrix multiply instead of three; minor efficiency gain            |
| Weight tying                    | Reduces parameters; token input/output representations share structure    |
| AdamW over Adam                 | Correct decoupled weight decay; better generalisation in practice         |
| Cosine LR schedule              | Smooth decay encourages convergence to flat minima                        |
| Checkpoint on val_loss improvement | Saves only the best model; avoids storing every epoch                 |
| `weights_only=True` in torch.load | Security: prevents arbitrary code execution from untrusted checkpoints |
| No sinusoidal positional encoding | Learned positional embeddings are simpler to implement; sufficient for short sequences |

---

## 11. Known Limitations

- **Synthetic data only**: The model trains on artificial text patterns and will not generalise to real-world language without a real corpus.
- **Small scale**: ~821K parameters and 2 epochs are sufficient for a demonstration but not production-quality language modelling.
- **No GPU multi-device support**: Single device (CPU or single GPU); no `DataParallel` or `DistributedDataParallel`.
- **No learning-rate warmup**: Pre-LN reduces but does not eliminate the benefit of warmup at larger scales.
- **Fixed vocabulary**: The 100-character printable ASCII vocabulary cannot represent Unicode text.
