"""
Dummy training data generation.

Generates character-level datasets from a mix of synthetic text patterns:
  - arithmetic expressions  (e.g. "3 + 7 = 10")
  - repeated phrase templates
  - simple word sequences
  - random ASCII sentences

Vocabulary: string.printable (100 printable ASCII characters) so all generated
text is human-readable.
"""

import random
import string
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from config import ModelConfig

# ---------------------------------------------------------------------------
# Vocabulary — 100 printable ASCII characters
# ---------------------------------------------------------------------------

VOCAB = string.printable          # '0-9a-zA-Z punctuation whitespace' = 100 chars
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR = {i: ch for i, ch in enumerate(VOCAB)}


# ---------------------------------------------------------------------------
# Text corpus generation
# ---------------------------------------------------------------------------

TEMPLATES = [
    "the cat sat on the mat",
    "hello world this is a test",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "a journey of a thousand miles begins with a single step",
    "to infinity and beyond",
    "once upon a time in a land far away",
    "the quick brown fox jumps over the lazy dog",
    "ask not what your country can do for you",
    "in the beginning was the word",
]

ADJECTIVES = ["small", "large", "fast", "slow", "bright", "dark", "warm", "cold", "old", "new"]
NOUNS = ["dog", "cat", "bird", "tree", "house", "car", "book", "river", "mountain", "star"]
VERBS = ["runs", "sits", "flies", "grows", "shines", "moves", "sleeps", "plays", "reads", "flows"]


def _arithmetic_sentence() -> str:
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    op = random.choice(["+", "-", "*"])
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    else:
        result = a * b
    return f"{a} {op} {b} = {result}"


def _simple_sentence() -> str:
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    verb = random.choice(VERBS)
    return f"the {adj} {noun} {verb}"


def _random_sentence(length: int = 40) -> str:
    chars = string.ascii_lowercase + " "
    return "".join(random.choice(chars) for _ in range(length))


def _template_sentence() -> str:
    base = random.choice(TEMPLATES)
    # Occasionally repeat the sentence twice for longer context
    if random.random() < 0.3:
        base = base + " " + base
    return base


def generate_corpus(num_sentences: int = 5000) -> str:
    """Build a large text string from synthetic sentences."""
    generators = [
        _arithmetic_sentence,
        _simple_sentence,
        _random_sentence,
        _template_sentence,
    ]
    lines = []
    for _ in range(num_sentences):
        gen = random.choice(generators)
        lines.append(gen())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Character-level tokenisation (printable ASCII, vocab_size = 100)
# ---------------------------------------------------------------------------

def encode(text: str) -> list[int]:
    # Unknown characters map to index of space
    space_idx = CHAR2IDX.get(" ", 0)
    return [CHAR2IDX.get(c, space_idx) for c in text]


def decode(tokens: list[int]) -> str:
    return "".join(IDX2CHAR.get(t, "") for t in tokens)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Sliding-window character-level dataset.

    Each sample is (input_ids, target_ids) where target_ids = input_ids shifted
    left by one position (next-token prediction).
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: ModelConfig):
    """
    Build train and validation DataLoaders from a freshly generated corpus.

    Returns:
        train_loader, val_loader, encode_fn, decode_fn
    """
    print("Generating dummy text corpus …")
    corpus = generate_corpus(num_sentences=6000)
    token_ids = encode(corpus)
    print(f"  Corpus length : {len(corpus):,} characters")
    print(f"  Token IDs     : {len(token_ids):,} tokens")

    dataset = TextDataset(token_ids, cfg.train_seq_len)
    print(f"  Dataset size  : {len(dataset):,} samples (seq_len={cfg.train_seq_len})")

    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    print(f"  Train batches : {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader, encode, decode


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig()
    train_loader, val_loader, enc, dec = get_dataloaders(cfg)
    x, y = next(iter(train_loader))
    print(f"\nSample batch  x: {x.shape}, y: {y.shape}")
    print(f"Decoded x[0]  : {repr(dec(x[0].tolist()))}")
