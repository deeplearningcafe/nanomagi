import torch
import torch.nn.functional as F


class KVCache:
    """
    KV Cache for Nanomagi. Matches PyTorch native attention layouts.
    """

    def __init__(
        self,
        batch_size,
        num_kv_heads,
        seq_len,
        head_dim,
        num_layers,
        device,
        dtype,
    ):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Store in (layers, B, H_kv, T, D) shape
        self.k_cache = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            seq_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.cache_seqlens = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )

    def reset(self):
        self.cache_seqlens.zero_()

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        assert self.get_pos() == 0, "KV cache is not empty"
        assert self.n_layers == other.n_layers
        assert self.n_kv_heads == other.n_kv_heads
        assert self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        # Expand batch size 1 to parallel sample batch size
        self.k_cache[:, :, :, :other_pos, :] = (
            other.k_cache[:, :, :, :other_pos, :]
            .expand(-1, self.batch_size, -1, -1, -1)
        )
        self.v_cache[:, :, :, :other_pos, :] = (
            other.v_cache[:, :, :, :other_pos, :]
            .expand(-1, self.batch_size, -1, -1, -1)
        )
        self.cache_seqlens.fill_(other_pos)


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    assert temperature >= 0.0, "Temperature cannot be negative."
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class RowState:

    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=None,
        temperature=1.0,
        top_k=None,
        seed=42,
    ):
        assert isinstance(tokens, list) and isinstance(
            tokens[0], int
        ), "Tokens must be a list of ints."
        device = self.model.get_device()
        dtype = self.model.transformer.wte.weight.dtype
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Prefill stage
        m = self.model.config
        kv_kwargs = {
            "num_kv_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)

        # Decode stage initialization
        kv_len = (
            (len(tokens) + max_tokens)
            if max_tokens is not None
            else m.max_position_embeddings
        )
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_len,
            device=device,
            dtype=dtype,
            **kv_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        num_generated = 0

        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            for i, state in enumerate(row_states):
                next_token = sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True

            yield token_column
            num_generated += 1

            ids = torch.tensor(
                token_column, dtype=torch.long, device=device
            ).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[
                :, -1, :
            ]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column in self.generate(tokens, num_samples, **kwargs):
            for i, token in enumerate(token_column):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
            if all(completed):
                break
        return results