import torch
import torch.nn as nn
from model.layers import MultiHeadAttention, FeedForward, RMSNorm
from model.config import ModelConfig

class VoxBlock(nn.Module):
    """A single Qwen-style Transformer Block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Attention layer (uses GQA from layers.py)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.n_kv_heads, config.dropout)
        # Feed-Forward Network
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        # Pre-normalization (RMSNorm before attention)
        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        # Pre-normalization (RMSNorm before FFN)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # Attention sub-layer with Residual Connection
        x = x + self.attn(self.norm1(x))
        # FFN sub-layer with Residual Connection
        x = x + self.ff(self.norm2(x))
        return x

class VoxModel(nn.Module):
    """The main Qwen-style Decoder-only Transformer Model."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Token Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # 2. Positional Embeddings (Trained Positional Encoding)
        # NOTE: A more advanced approach uses RoPE, but trained embeddings are simpler.
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model) 
        
        # 3. Stacked Transformer Blocks
        self.layers = nn.ModuleList([QwenBlock(config) for _ in range(config.n_layers)])
        
        # 4. Final Normalization Layer
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        
        # 5. Language Modeling Head (Maps d_model back to vocab_size)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        
        # 1. Embeddings + Position
        # Create positions tensor [0, 1, 2, ..., T-1]
        positions = torch.arange(T, device=x.device).unsqueeze(0) 
        
        # Sum token and position embeddings
        h = self.token_emb(x) + self.pos_emb(positions)
        
        # 2. Pass through all layers
        for layer in self.layers:
            h = layer(h)
            
        # 3. Final Norm and Logits
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits