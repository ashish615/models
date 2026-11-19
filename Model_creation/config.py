"""Model hyperparameters and training configuration."""

class ModelConfig:
    # Vocabulary
    vocab_size: int = 100          # printable ASCII only (string.printable = 100 chars)

    # Architecture
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512             # feed-forward hidden size (4x embed_dim)
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 2
    grad_clip: float = 1.0

    # Data
    train_seq_len: int = 64        # context window for training samples
    num_train_samples: int = 2000
    num_val_samples: int = 400

    # Generation
    temperature: float = 0.8
    top_k: int = 40

    # Paths
    checkpoint_path: str = "model_checkpoint.pt"
