import os
import logging
import json
import fsspec.spec
import fsspec.utils
import pyarrow
import pyarrow.dataset as pad
from datasets import load_dataset, interleave_datasets
from datasets.distributed import split_dataset_by_node

from nanomagi.utils import get_dist_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mixed_streaming_dataset(
    stage: int = 1,
    split: str = "train",
    num_val_samples: int = 10000,
    seed: int = 42,
):
    """
    Loads, mixtures, and cleanly splits Japanese streaming datasets.
    Using split="val" grabs the first num_val_samples deterministically.
    Using split="train" skips them to avoid data contamination (leakage).
    """
    logger.info("Initializing multi-stage Japanese pretraining streams...")

    # Configure global fsspec block sizes to 128 MiB to minimize API requests
    fsspec.spec.AbstractBufferedFile.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024
    fsspec.utils.DEFAULT_BLOCK_SIZE = 128 * 1024 * 1024

    # Caching options passed to remote fsspec filesystems (JSONL, text, etc.)
    storage_options = {
        "block_size": 128 * 1024 * 1024,
        "cache_type": "readahead",
    }

    # Prefetch scan options specifically optimized for Parquet-based streaming
    fragment_scan_options = pad.ParquetFragmentScanOptions(
        cache_options=pyarrow.CacheOptions(
            prefetch_limit=2,
            range_size_limit=128 << 20,
        ),
    )

    fw_ds = load_dataset(
        "hotchpotch/fineweb-2-edu-japanese",
        name="default",
        split="train",
        streaming=True,
        storage_options=storage_options,
        fragment_scan_options=fragment_scan_options,
    ).select_columns(["text"])

    abeja_cc_ds = load_dataset(
        "kajuma/ABEJA-CC-JA-edu",
        name="10%",
        split="train",
        streaming=True,
        storage_options=storage_options,
        fragment_scan_options=fragment_scan_options,
    ).select_columns(["content"])

    wiki_ds = load_dataset(
        "izumi-lab/wikipedia-ja-20230720",
        split="train",
        streaming=True,
        storage_options=storage_options,
    ).select_columns(["text"])

    aozora_ds = load_dataset(
        "globis-university/aozorabunko-clean",
        split="train",
        streaming=True,
        storage_options=storage_options,
    )

    # Filter for modern Japanese
    aozora_ds = aozora_ds.filter(
        lambda row: row["meta"].get("文字遣い種別") == "新字新仮名"
    ).select_columns(["text"])

    # Assign mixture coefficients
    if stage == 1:
        probs = [0.58, 0.30, 0.10, 0.02]
    elif stage == 2:
        probs = [0.30, 0.20, 0.35, 0.15]
    else:
        raise ValueError(f"Invalid dataset mixture stage: {stage}")

    mixed_dataset = interleave_datasets(
        [fw_ds, abeja_cc_ds, wiki_ds, aozora_ds],
        probabilities=probs,
        seed=seed,
    )

    # Perform disjoint split using take/skip before DDP sharding
    if split == "val":
        mixed_dataset = mixed_dataset.take(num_val_samples)
        logger.info(f"Validation split active: taken {num_val_samples} samples.")
    elif split == "train":
        mixed_dataset = mixed_dataset.skip(num_val_samples)
        logger.info(f"Train split active: skipped first {num_val_samples} samples.")
    else:
        raise ValueError(f"Invalid split: {split}")

    # Shard across distributed ranks if using multi-GPU DDP training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_world_size > 1 and split == "train":
        logger.info(f"DDP active. Sharding stream on rank {ddp_rank}...")
        mixed_dataset = split_dataset_by_node(
            mixed_dataset, rank=ddp_rank, world_size=ddp_world_size
        )

    return mixed_dataset


def build_static_validation_set(
    output_path: str,
    stage: int = 1,
    num_samples: int = 10000,
    seed: int = 42,
):
    """
    Serializes the deterministic validation split to a local JSONL file.
    """
    logger.info(f"Building static validation dataset to {output_path}...")
    dataset = get_mixed_streaming_dataset(
        stage=stage, split="val", num_val_samples=num_samples, seed=seed
    )

    val_data = []
    for i, example in enumerate(dataset):
        text = example.get("text", "")
        if text:
            val_data.append({"text": text})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Successfully wrote {len(val_data)} validation samples to local disk.")


def load_static_validation_dataset(val_path: str):
    """
    Loads the static local validation dataset from JSONL.
    """
    import json

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    dataset = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def get_sft_dataset(split="train", num_samples=15000, seed=42):
    """
    Loads and preprocesses the aya-ja-evol-instruct-calm3-dpo-masked dataset.
    Given resource constraints and to preserve pretraining knowledge, we train
    only on 10k-20k samples for SFT.
    """
    ds = load_dataset(
        "weblab-GENIAC/aya-ja-evol-instruct-calm3-dpo-masked",
        split="train",
    )
    ds = ds.shuffle(seed=seed)

    def preprocess_example(example):
        messages = []
        for msg in example["prompt"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        messages.append({
            "role": "assistant",
            "content": example["chosen"]
        })
        return {"messages": messages}

    ds = ds.map(preprocess_example)

    # Disjoint split
    val_size = num_samples // 4
    if split == "train":
        ds = ds.select(range(val_size, val_size + num_samples))
    elif split == "val":
        ds = ds.select(range(0, val_size))
    else:
        raise ValueError(f"Invalid split: {split}")

    return ds