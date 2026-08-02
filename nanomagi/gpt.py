import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 100000.0
    use_checkpointing: bool = False


class RotaryEmbedding(nn.Module):
    """
    1D Rotary Position Embedding (RoPE) module.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 100000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        steps = torch.arange(0, self.dim, 2).float()
        inv_freq = 1.0 / (self.base ** (steps / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device="cpu")

    def _set_cos_sin_cache(self, seq_len: int, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(t.device))

        # [seq_len, dim // 2] -> [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # [seq_len, dim] -> [1, seq_len, 1, dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, seq_len: int):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        return (
            self.cos_cached[:, :seq_len].to(x.device, dtype=x.dtype),
            self.sin_cached[:, :seq_len].to(x.device, dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: [B, T, H, D]
    # cos, sin shape: [1, T, 1, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU)
    """

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CausalSelfAttention(nn.Module):
    """
    GQA-capable Causal Self-Attention with QK-Normalization and 1D RoPE.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        # Unified QKV
        qkv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.c_attn = nn.Linear(self.n_embd, qkv_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # TODO: remove params?
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q_size = self.n_head * self.head_dim
        kv_size = self.n_kv_head * self.head_dim

        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat K, V heads
        if self.n_head != self.n_kv_head:
            num_queries_per_kv = self.n_head // self.n_kv_head
            k = torch.repeat_interleave(k, num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, num_queries_per_kv, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    """
    Standard pre-normalized decoder Block.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config.n_embd, config.intermediate_size)
        self.use_checkpointing = config.use_checkpointing

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        if self.use_checkpointing:
            mlp_out = torch.utils.checkpoint.checkpoint(
                self.mlp,
                self.post_attention_layernorm(x),
                use_reentrant=False,
            )
        else:
            mlp_out = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out

        return x


class GPT(nn.Module):
    """
    Vanilla Autoregressive LLM
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # TODO: norm after embeds?
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.RMSNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize RoPE 1D
        self.rotary_emb = RotaryEmbedding(
            dim=config.n_embd // config.n_head,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.apply(self._init_weights)

        # Zero projections, scaled uniform inputs
        for _, module in self.named_modules():
            if isinstance(module, CausalSelfAttention):
                std = 0.5 * (module.n_embd**-0.5)
                bound = (3.0**0.5) * std
                nn.init.uniform_(module.c_attn.weight, -bound, bound)
                nn.init.zeros_(module.c_proj.weight)
            elif isinstance(module, SwiGLU):
                std = 0.5 * (module.w1.in_features**-0.5)
                bound = (3.0**0.5) * std
                nn.init.uniform_(module.w1.weight, -bound, bound)
                nn.init.uniform_(module.w2.weight, -bound, bound)
                nn.init.zeros_(module.w3.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_device(self):
        return self.transformer.wte.weight.device

    def non_embed_params(self):
        total_params = 0
        total_params += sum(
            p.numel() for p in self.transformer["h"].parameters() if p.requires_grad
        )
        total_params += sum(
            p.numel() for p in self.transformer["ln_f"].parameters() if p.requires_grad
        )
        total_params += sum(
            p.numel() for p in self.lm_head.parameters() if p.requires_grad
        )
        return total_params

    def embed_params(self):
        return sum(
            p.numel() for p in self.transformer["wte"].parameters() if p.requires_grad
        )

    def estimate_flops(self):
        """Return estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l = self.config.n_layer
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.max_position_embeddings
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()

        # Fetch position-aligned rotary embeddings
        cos, sin = self.rotary_emb(idx, T)

        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # Softcapping
        # Logit softcapping in float32 for training stability
        logits = logits.float()
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.to(x.dtype)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Simple autoregressive next-token generator.
        """
        assert isinstance(tokens, list)
        device = self.get_device()

        generator = None
        if temperature > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=generator)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()

    @torch.inference_mode()
    def generate_chat(
        self,
        tokens,
        max_tokens,
        temperature=1.0,
        top_k=None,
        seed=42,
        eos_token_id=None,
    ):
        """
        Autoregressive generator for chat that stops when eos_token_id is met.
        """
        assert isinstance(tokens, list)
        device = self.get_device()

        generator = None
        if temperature > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(
                    probs, num_samples=1, generator=generator
                )
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_ids.item()
            if eos_token_id is not None and token_id == eos_token_id:
                break

            ids = torch.cat((ids, next_ids), dim=1)
            yield token_id