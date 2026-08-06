"""
A number of functions that help with evaluating a base model.
"""

import math
import torch
import torch.distributed as dist
from nanomagi.dataset import load_static_validation_dataset

def get_token_bytes(tokenizer, device="cpu"):
    """
    Computes a 1D tensor of byte lengths for all tokens in vocabulary.
    Special tokens contribute 0 bytes.
    """
    vocab_size = tokenizer.get_vocab_size()
    special_ids = {
        tokenizer.encode_special(s)
        for s in tokenizer.get_special_tokens()
        if tokenizer.encode_special(s) is not None
    }
    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
            token_bytes.append(num_bytes)
    return torch.tensor(token_bytes, dtype=torch.int32, device=device)



@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Evaluates Bits Per Byte (BPB) metric over batches.
    `token_bytes` can be either a 1D Tensor or a Tokenizer wrapper.
    """
    device = model.get_device()
    if not isinstance(token_bytes, torch.Tensor):
        token_bytes = get_token_bytes(token_bytes, device=device)
    else:
        token_bytes = token_bytes.to(device)

    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction="none")  # (B, T)
        loss2d = loss2d.view(-1)  # flatten
        y = y.view(-1)  # flatten
        if (
            y.int() < 0
        ).any():  # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid, token_bytes[y_safe], torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("inf")
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


@torch.no_grad()
def compute_perplexity(
    model, tokenizer, val_path, device, token_bytes=None, max_len=2048
):
    """
    Computes Cross-Entropy Loss, Perplexity, and Bits Per Byte (BPB)
    on local validation holdout dataset.
    """
    model.eval()
    dataset = load_static_validation_dataset(val_path)

    if token_bytes is None or not isinstance(token_bytes, torch.Tensor):
        token_bytes = get_token_bytes(tokenizer, device=device)
    else:
        token_bytes = token_bytes.to(device)

    total_loss = 0.0
    total_tokens = 0
    total_nats = 0.0
    total_bytes = 0

    bos_id = tokenizer.get_bos_token_id()

    for item in dataset:
        text = item.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text, prepend=bos_id)[:max_len]
        if len(tokens) < 2:
            continue

        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

        logits = model(input_ids)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )

        y = targets.view(-1)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )

        total_loss += token_losses.sum().item()
        total_tokens += targets.numel()

        valid_bytes_mask = num_bytes > 0
        total_nats += (token_losses * valid_bytes_mask).sum().item()
        total_bytes += num_bytes.sum().item()

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    bpb = (
        total_nats / (math.log(2) * total_bytes)
        if total_bytes > 0
        else float("inf")
    )

    return ppl, mean_loss, bpb

@torch.no_grad()
def evaluate_sft_val_loss(
    model, val_loader, eval_steps, device, ignore_index=-1
):
    model.eval()
    total_loss = 0.0
    total_active_tokens = 0
    steps_conducted = 0

    for _ in range(eval_steps):
        try:
            inputs, targets, _ = next(val_loader)
        except StopIteration:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)

        shift_logits = logits.view(-1, logits.size(-1))
        shift_targets = targets.view(-1)

        active_mask = shift_targets != ignore_index
        num_active_tokens = active_mask.sum().item()

        if num_active_tokens == 0:
            continue

        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="sum"
        )
        sum_loss = loss_fct(shift_logits, shift_targets)

        total_loss += sum_loss.item()
        total_active_tokens += num_active_tokens
        steps_conducted += 1

    t_loss = torch.tensor(total_loss, device=device)
    t_tokens = torch.tensor(
        total_active_tokens, device=device, dtype=torch.float32
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_tokens, op=dist.ReduceOp.SUM)

    total_loss_all = t_loss.item()
    total_tokens_all = t_tokens.item()

    if total_tokens_all == 0:
        return float("inf"), float("inf")

    mean_val_loss = total_loss_all / total_tokens_all
    val_ppl = (
        math.exp(mean_val_loss) if mean_val_loss < 20 else float("inf")
    )

    model.train()
    return mean_val_loss, val_ppl
