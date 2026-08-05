import torch
from nanomagi.dataset import get_mixed_streaming_dataset
from nanomagi.dataset import get_sft_dataset
from nanomagi.utils import get_dist_info




def _document_batches(iterable_dataset, tokenizer_batch_size=128):
    """
    Infinite generator yielding lists of document strings of size
    tokenizer_batch_size from our mixed streaming stream.
    """
    epoch = 1
    while True:
        # Update epoch seed dynamically if supported to reshuffle
        if hasattr(iterable_dataset, "set_epoch"):
            iterable_dataset.set_epoch(epoch)

        batch = []
        for example in iterable_dataset:
            text = example.get("text") or example.get("content")
            if text:
                batch.append(text)
                if len(batch) == tokenizer_batch_size:
                    yield batch, epoch
                    batch = []

        if batch:
            yield batch, epoch

        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    B,
    T,
    split="train",
    stage: int = 1,
    seed: int = 42,
    num_val_samples: int = 10000,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    buffer_size=1000,
):
    """
    Causal autoregressive dataloader featuring BOS-aligned Best-Fit Packing.
    Fully optimized to stream remote datasets without disk footprint.
    """
    if split != "train":
        raise ValueError("Only train split streaming is supported.")

    row_capacity = T + 1

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    base_seed = seed - ddp_rank if ddp else seed

    mixed_ds = get_mixed_streaming_dataset(
        stage=stage,
        split=split,
        num_val_samples=num_val_samples,
        seed=base_seed,
    )
    batches = _document_batches(mixed_ds, tokenizer_batch_size)

    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        # Tokenize batch using the tokenizer's multicore thread pool
        token_lists = tokenizer.encode(
            doc_batch, prepend=bos_token, num_threads=tokenizer_threads
        )
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate contiguous inputs and targets to maximize memory bandwidth
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)

    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # BOS-Aligned Best-Fit selection logic
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    pos += doc_len
                else:
                    # Crop the shortest buffered item to fill the space
                    shortest_idx = min(
                        range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])
                    )
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        # Single HtoD CPU pinned transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)

        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields for backward compatibility."""
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs)
    for inputs, targets, _ in loader:
        yield inputs, targets

def sft_data_loader_bos_bestfit(
    tokenizer,
    B,
    T,
    split="train",
    seed=42,
    device="cuda",
    buffer_size=100,
):
    """
    SFT Dataloader with BOS-aligned Best-Fit Packing and loss masking.
    Computes loss only on assistant part and masks padding.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Recover base seed across DDP ranks to keep dataset splits consistent
    base_seed = seed - ddp_rank if ddp else seed

    dataset = get_sft_dataset(split=split, seed=base_seed)

    dataset_size = len(dataset)
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank
    consumed = ddp_rank
    epoch = 1

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(
                conversation, max_tokens=T
            )
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows = []
        mask_rows = []
        row_lengths = []
        for _ in range(B):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break

            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long)
        inputs = batch_tensor[:, :-1].to(
            device=device, dtype=torch.long, non_blocking=use_cuda
        ).contiguous()
        targets = batch_tensor[:, 1:].to(
            device=device, dtype=torch.long, non_blocking=use_cuda
        ).contiguous()

        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        state_dict = {"epoch": epoch, "consumed": consumed}
        yield inputs, targets, state_dict
