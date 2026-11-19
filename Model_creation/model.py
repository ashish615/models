"""
GPT-like decoder-only transformer built entirely from scratch using PyTorch primitives.

Architecture:
  TokenEmbedding + PositionalEmbedding
  → N x TransformerBlock (causal self-attention + FFN + LayerNorm)
  → LM Head (linear projection to vocab logits)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Fused QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask: lower-triangular so token i can only attend to 0..i
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv(x)                         # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)            # each (B, T, C)

        # Reshape to (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                             # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Linear."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: LN → Attention → residual, LN → FFN → residual."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """
    Small GPT-style language model.

    Args:
        vocab_size:   Number of token types.
        embed_dim:    Token/position embedding dimension.
        num_heads:    Number of attention heads per block.
        num_layers:   Number of stacked TransformerBlocks.
        ffn_dim:      Hidden size inside the FFN.
        max_seq_len:  Maximum sequence length (also the positional embedding table size).
        dropout:      Dropout probability applied throughout.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share token embedding weights with the LM head
        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx:     (B, T) integer token indices.
            targets: (B, T) shifted token indices for teacher-forcing loss.

        Returns:
            logits: (B, T, vocab_size)
            loss:   cross-entropy scalar if targets provided, else None.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.embed_drop(self.token_embed(idx) + self.pos_embed(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressively generate tokens given a prompt.

        Args:
            idx:            (B, T) prompt token indices.
            max_new_tokens: Number of tokens to generate.
            temperature:    Sampling temperature (< 1 sharpens, > 1 flattens distribution).
            top_k:          If set, restricts sampling to the top-k logits.

        Returns:
            (B, T + max_new_tokens) token indices.
        """
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
