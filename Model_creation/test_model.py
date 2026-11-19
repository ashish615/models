"""
Test suite for MiniLLM.

Tests are grouped into three sections:
  1. Unit tests  – architecture / forward-pass correctness
  2. Training tests – loss decreases over a few steps on a tiny batch
  3. Generation tests – output shape, temperature, top-k, perplexity
"""

import math
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import MiniLLM
from data import encode, decode, generate_corpus, TextDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(cfg: ModelConfig = None) -> MiniLLM:
    if cfg is None:
        cfg = ModelConfig()
    return MiniLLM(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=0.0,   # deterministic for testing
    )


def _dummy_batch(batch_size: int = 2, seq_len: int = 16, vocab_size: int = None):
    if vocab_size is None:
        vocab_size = ModelConfig().vocab_size
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y


def _compute_perplexity(model: MiniLLM, token_ids: list, seq_len: int = 64) -> float:
    model.eval()
    data = torch.tensor(token_ids, dtype=torch.long)
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(data) - seq_len, seq_len):
            chunk = data[i: i + seq_len + 1]
            if len(chunk) < seq_len + 1:
                break
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            _, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / max(num_batches, 1)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# 1. Unit tests
# ---------------------------------------------------------------------------

def test_output_shape():
    cfg = ModelConfig()
    model = _make_model(cfg)
    x, y = _dummy_batch(batch_size=4, seq_len=32, vocab_size=cfg.vocab_size)
    logits, loss = model(x, y)

    assert logits.shape == (4, 32, cfg.vocab_size), \
        f"Expected (4, 32, {cfg.vocab_size}), got {logits.shape}"
    assert loss is not None, "Loss should not be None when targets are provided"
    assert loss.item() > 0, "Loss must be positive"
    print(f"[PASS] test_output_shape  | logits={logits.shape}, loss={loss.item():.4f}")


def test_no_target_no_loss():
    model = _make_model()
    x = torch.randint(0, 256, (2, 16))
    logits, loss = model(x, targets=None)

    assert logits.shape == (2, 16, 256)
    assert loss is None, "Loss should be None when targets not given"
    print(f"[PASS] test_no_target_no_loss | logits={logits.shape}")


def test_parameter_count():
    model = _make_model()
    n = model.num_parameters()
    assert n > 0, "Model should have trainable parameters"
    print(f"[PASS] test_parameter_count | {n:,} parameters")


def test_weight_tying():
    model = _make_model()
    assert model.token_embed.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "Token embedding and LM head weights should be tied"
    print(f"[PASS] test_weight_tying")


def test_causal_mask():
    """Verify that token i cannot attend to token i+1 (causality)."""
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()

    seq_len = 10
    x = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # Baseline logits
    with torch.no_grad():
        logits_orig, _ = model(x)

    # Overwrite the last token and recompute; logits for earlier positions must be unchanged
    x_modified = x.clone()
    x_modified[0, -1] = (x[0, -1].item() + 1) % cfg.vocab_size
    with torch.no_grad():
        logits_mod, _ = model(x_modified)

    diff = (logits_orig[0, :-1] - logits_mod[0, :-1]).abs().max().item()
    assert diff < 1e-5, f"Causal mask violated: max logit diff={diff:.2e}"
    print(f"[PASS] test_causal_mask | max_diff={diff:.2e}")


def test_sequence_length_boundary():
    cfg = ModelConfig()
    model = _make_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, _ = model(x)
    assert logits.shape[1] == cfg.max_seq_len
    print(f"[PASS] test_sequence_length_boundary | seq_len={cfg.max_seq_len}")


def test_exceeds_max_seq_len_raises():
    cfg = ModelConfig()
    model = _make_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    try:
        model(x)
        print(f"[FAIL] test_exceeds_max_seq_len_raises | expected AssertionError")
    except AssertionError:
        print(f"[PASS] test_exceeds_max_seq_len_raises")


# ---------------------------------------------------------------------------
# 2. Training tests
# ---------------------------------------------------------------------------

def test_loss_decreases():
    """Loss should drop noticeably after 50 gradient steps on a fixed tiny batch."""
    cfg = ModelConfig()
    cfg.dropout = 0.0
    model = _make_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x, y = _dummy_batch(batch_size=8, seq_len=32, vocab_size=cfg.vocab_size)

    model.train()
    _, loss_start = model(x, y)
    loss_start = loss_start.item()

    for _ in range(50):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, loss_end = model(x, y)
    loss_end = loss_end.item()

    assert loss_end < loss_start, \
        f"Loss should decrease: start={loss_start:.4f}, end={loss_end:.4f}"
    print(f"[PASS] test_loss_decreases | {loss_start:.4f} -> {loss_end:.4f}")


def test_gradient_flow():
    """All parameters should receive gradients after a backward pass."""
    model = _make_model()
    x, y = _dummy_batch(batch_size=2, seq_len=16)
    _, loss = model(x, y)
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"Missing gradients: {missing}"
    print(f"[PASS] test_gradient_flow | all {model.num_parameters():,} params have gradients")


# ---------------------------------------------------------------------------
# 3. Generation tests
# ---------------------------------------------------------------------------

def test_generate_shape():
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    max_new = 20
    out = model.generate(prompt, max_new_tokens=max_new)
    assert out.shape == (1, 5 + max_new), f"Expected (1, 25), got {out.shape}"
    print(f"[PASS] test_generate_shape | output={out.shape}")


def test_generate_in_vocab():
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=30, temperature=1.0)
    assert out.max().item() < cfg.vocab_size, "Generated token exceeds vocab_size"
    assert out.min().item() >= 0, "Generated token is negative"
    print(f"[PASS] test_generate_in_vocab")


def test_generate_temperature():
    """Lower temperature should produce more deterministic (less varied) output."""
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()
    torch.manual_seed(42)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))

    outputs_low = set()
    outputs_high = set()
    for _ in range(10):
        out_low = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.1)
        out_high = model.generate(prompt.clone(), max_new_tokens=10, temperature=2.0)
        outputs_low.add(tuple(out_low[0].tolist()))
        outputs_high.add(tuple(out_high[0].tolist()))

    # Low temp should be more repetitive (fewer unique sequences)
    assert len(outputs_low) <= len(outputs_high), \
        "Low temperature should yield less diversity than high temperature"
    print(f"[PASS] test_generate_temperature | low_temp unique={len(outputs_low)}, high_temp unique={len(outputs_high)}")


def test_generate_top_k():
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=15, temperature=1.0, top_k=5)
    assert out.shape[1] == 4 + 15
    print(f"[PASS] test_generate_top_k | output={out.shape}")


def test_perplexity_on_corpus():
    """Perplexity on a small corpus should be finite and below vocab_size."""
    cfg = ModelConfig()
    model = _make_model(cfg)
    corpus = generate_corpus(num_sentences=200)
    tokens = encode(corpus)
    ppl = _compute_perplexity(model, tokens, seq_len=cfg.train_seq_len)
    assert math.isfinite(ppl), "Perplexity is not finite"
    # Untrained model can approach vocab_size; cap at 4x as a sanity bound
    assert ppl < cfg.vocab_size * 4, f"Perplexity {ppl:.1f} is unreasonably high (>{cfg.vocab_size * 4})"
    print(f"[PASS] test_perplexity_on_corpus | perplexity={ppl:.2f}")


def test_text_generation_from_prompt():
    """End-to-end: encode a text prompt, generate, decode."""
    cfg = ModelConfig()
    model = _make_model(cfg)
    model.eval()

    prompt_text = "hello world"
    prompt_ids = torch.tensor([encode(prompt_text)], dtype=torch.long)
    out_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.8, top_k=40)
    out_text = decode(out_ids[0].tolist())

    assert out_text.startswith(prompt_text), "Generated text should start with the prompt"
    assert len(out_text) == len(prompt_text) + 30
    print(f"[PASS] test_text_generation_from_prompt | '{out_text}'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # Unit tests
    test_output_shape,
    test_no_target_no_loss,
    test_parameter_count,
    test_weight_tying,
    test_causal_mask,
    test_sequence_length_boundary,
    test_exceeds_max_seq_len_raises,
    # Training tests
    test_loss_decreases,
    test_gradient_flow,
    # Generation tests
    test_generate_shape,
    test_generate_in_vocab,
    test_generate_temperature,
    test_generate_top_k,
    test_perplexity_on_corpus,
    test_text_generation_from_prompt,
]


if __name__ == "__main__":
    print("=" * 60)
    print("MiniLLM Test Suite")
    print("=" * 60)

    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__} | {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    print("=" * 60)
