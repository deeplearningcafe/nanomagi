import torch
from nanomagi.dataset import get_mixed_streaming_dataset


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
            text = example.get("text")
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
    mixed_ds = get_mixed_streaming_dataset(
        stage=stage,
        split=split,
        num_val_samples=num_val_samples,
        seed=seed,
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
