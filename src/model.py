"""A small GPT-style transformer (RMSNorm, RoPE, GQA, SwiGLU, weight tying).

This is the instrument the Gate-projectable weights live in. Deliberately
compact and dependency-free so the full pipeline runs on a laptop CPU and the
mechanism stays readable. A 4-layer / 256-wide config is 4.17M parameters.
"""
import torch, torch.nn as nn


class Config:
    def __init__(self, vocab_size, d_model=256, n_layer=4, n_head=4,
                 n_kv_head=2, block_size=16, ffn_mult=4, dropout=0.0,
                 tie_weights=False):
        self.vocab_size = vocab_size; self.d_model = d_model
        self.n_layer = n_layer; self.n_head = n_head; self.n_kv_head = n_kv_head
        self.block_size = block_size; self.ffn_mult = ffn_mult
        self.dropout = dropout; self.tie_weights = tie_weights


class RMSNorm(nn.Module):
    """Root-mean-square layer norm; cheaper than LayerNorm, common in modern LLMs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope(cfg, dtype=torch.float32, device="cpu"):
    """Rotary position embeddings: cos/sin per position, sized for one head (d_head/2)."""
    d_head = cfg.d_model // cfg.n_head
    dim = d_head
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device)
                                   .float() / dim))
    pos = torch.arange(cfg.block_size, device=device).float()
    freqs = pos[:, None] * inv_freq[None, :]           # [T, d_head/2]
    cos = freqs.cos().to(dtype)                        # [T, d_head/2]
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    """Rotate query/key halves. x: [B, T, H, d_head], cos/sin: [T, d_head/2]"""
    d_head = x.shape[-1]
    x1 = x[..., : d_head // 2]
    x2 = x[..., d_head // 2:]
    c = cos[None, : x.shape[1], None, :]
    s = sin[None, : x.shape[1], None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_head * self.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_head * self.d_head, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.d_head)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head)
        q = apply_rope(q, cos, sin).transpose(1, 2)    # [B, H, T, d_head]
        k = apply_rope(k, cos, sin).transpose(1, 2)    # [B, n_kv, T, d_head]
        # GQA: expand KV to match query heads
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin))
        return x + self.dropout(self.mlp(self.norm2(x)))


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        cos, sin = precompute_rope(cfg)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # WHY: tying replaces the Linear head's calibrated init (±1/sqrt(d))
            # with the embedding's native N(0,1) rows — logits explode and the
            # model can't fit. Standard fix: re-init the (now shared) matrix at
            # head scale so it behaves like a proper tied head (GPT-2/NMT style).
            bound = 1.0 / (cfg.d_model ** 0.5)
            nn.init.uniform_(self.tok.weight, -bound, bound)
            self.head.weight = self.tok.weight  # share embed/head after rescaling

    def forward(self, idx, activations=None):
        T = idx.shape[1]
        x = self.tok(idx)
        cos = self.cos[:T].to(x.dtype)
        sin = self.sin[:T].to(x.dtype)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin)
            if activations is not None:
                activations[i] = x.detach().norm(dim=-1)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
