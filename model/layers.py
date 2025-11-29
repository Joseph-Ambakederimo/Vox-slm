import torch
import torch.nn as nn
import torch.nn.functional as F

# rmsnorm used by qwen3
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.scale

# feedforward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Using SiLU (Swish) activation, common in modern LLMs
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# multi-head attention with grouped key/value heads (Grouped Query Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        # 1. Project and reshape Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # B, H_q, T, D_h
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2) # B, H_kv, T, D_h
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2) # B, H_kv, T, D_h

        # 2. Grouped Query Attention (GQA): Repeat KV heads to match Q heads
        if self.n_heads != self.n_kv_heads:
            # Replicate K and V heads to match the number of Q heads
            # (1, num_reps, 1, 1) repeats across the head dimension
            num_reps = self.n_heads // self.n_kv_heads
            k = k.repeat(1, num_reps, 1, 1)
            v = v.repeat(1, num_reps, 1, 1)
        
        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        # 4. CRITICAL: Causal Masking (for next-token prediction)
        # Creates a triangular matrix where the upper triangle (future tokens) is -1e9
        mask = torch.full((T, T), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        
        scores = scores + mask # Broadcasts across batch and head dimensions

        # 5. Softmax and Dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 6. Weighted sum of Values and Concatenation
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # B, T, C

        # 7. Final Linear Projection
        return self.out_proj(out)