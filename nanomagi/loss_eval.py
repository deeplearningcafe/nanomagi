"""
A number of functions that help with evaluating a base model.
"""

import math
import torch
import torch.distributed as dist
from nanomagi.dataset import load_static_validation_dataset


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
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
def compute_perplexity(model, tokenizer, val_path, device, max_len=2048):
    """
    Task 1: Compute Cross-Entropy Loss and Perplexity on local validation holdout.
    """
    model.eval()
    dataset = load_static_validation_dataset(val_path)

    total_loss = 0.0
    total_tokens = 0
    bos_id = tokenizer.get_bos_token_id()

    with torch.no_grad():
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
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    return ppl, mean_loss

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
