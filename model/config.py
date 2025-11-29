# import dataclasses

# @dataclasses.dataclass
# class ModelConfig:
#     # Model Architecture Parameters
#     vocab_size: int = 50000        # Must match your tokenizer's vocab_size
#     max_seq_len: int = 512        # T: Block Size / Context Length
#     n_layers: int = 6             # Number of Transformer blocks
#     d_model: int = 768            # C: Embedding dimension (must be divisible by n_heads)
#     n_heads: int = 12             # Number of attention heads
#     n_kv_heads: int = 4           # Number of Key/Value heads (for Grouped Query Attention)
#     d_ff: int = 3072              # Feed-Forward hidden size (often 4 * d_model)
#     dropout: float = 0.1          # Dropout rate for attention and FFN
#     rms_norm_eps: float = 1e-6    # Epsilon for RMSNorm

#     # Training Parameters
#     batch_size: int = 8
#     gradient_accumulation_steps: int = 4
#     muon_lr: float = 3e-4         # Learning Rate for Muon (or AdamW)
#     max_steps: int = 10000        # Total number of training steps
#     eval_every: int = 100         # Evaluate and Checkpoint every N steps
#     use_amp: bool = True          # Use Automatic Mixed Precision (requires CUDA)
#     num_documents: int = 20000    # Number of documents to load for dataset

import dataclasses

@dataclasses.dataclass
class ModelConfig:
    # Model Architecture Parameters
    vocab_size: int = 50000        # Must match your tokenizer's vocab_size
    max_seq_len: int = 512         # T: Block Size / Context Length
    n_layers: int = 6              # Number of Transformer blocks
    d_model: int = 768             # C: Embedding dimension (must be divisible by n_heads)
    n_heads: int = 12         # change to 12     # Number of attention heads
    n_kv_heads: int = 4            # Number of Key/Value heads (for Grouped Query Attention)
    d_ff: int = 3072               # Feed-Forward hidden size (often 4 * d_model)
    dropout: float = 0.1           # Dropout rate for attention and FFN
    rms_norm_eps: float = 1e-6     # Epsilon for RMSNorm

    # Training Parameters
    batch_size: int = 8            # Physical batch size
    gradient_accumulation_steps: int = 4 # Effective batch size = batch_size * this value
    muon_lr: float = 3e-4          # Learning Rate for Optimizer
    max_steps: int = 10000         # Total number of training steps
    eval_every: int = 100          # Evaluate and Checkpoint every N steps
    use_amp: bool = True           # Use Automatic Mixed Precision (requires CUDA)
    num_documents: int = 20000     # Number of documents to load for the main dataset