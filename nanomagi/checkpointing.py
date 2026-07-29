import os
import json
import logging
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


def save_checkpoint(
    output_dir: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    config: dict = None,
):
    """
    Saves model checkpoint weights, optimizer, scheduler, EMA,
    config.json (HuggingFace architecture style), and vocab.json.
    """

    save_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Cast only floating-point tensors to bfloat16 to halve disk size
    clean_state_dict = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in clean_state_dict.items()
    }

    model_path = os.path.join(save_dir, "model.safetensors")
    save_file(clean_state_dict, model_path)

    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    # 4. Save Config JSON (HuggingFace Style)
    if config is not None:
        hf_config = {
            "hidden_dim": config.get("hidden_dim", 128),
            "num_layers": config.get("num_layers", 3),
            "max_seq_len": config.get("max_seq_len", 16),
        }

        # Include other serializable configuration items
        for k, v in config.items():
            if k not in hf_config and isinstance(
                v, (int, float, str, bool, list, dict)
            ):
                hf_config[k] = v

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)

    logging.info(f"Checkpoint saved successfully at {save_dir}")


def load_checkpoint_config(checkpoint_dir: str) -> dict:
    """Loads config.json from checkpoint directory if present."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module = None,
    optimizer=None,
    scheduler=None,
) -> int:
    """
    Loads states from a checkpoint directory.
    Returns the resumed start step.
    """
    logging.info(f"Loading checkpoint from {checkpoint_dir}")

    strict = True
    if model is not None:
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            sanitized_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(sanitized_dict, strict=strict)

    if optimizer is not None:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu")
            if isinstance(optimizer, dict) and isinstance(opt_state, dict):
                for k, opt in optimizer.items():
                    if k in opt_state:
                        opt.load_state_dict(opt_state[k])
            else:
                optimizer.load_state_dict(opt_state)

    if scheduler is not None:
        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    start_step = 0
    parts = os.path.normpath(checkpoint_dir).split(os.sep)
    for part in parts:
        if part.startswith("step_"):
            try:
                start_step = int(part.split("_")[1])
            except ValueError:
                pass

    logging.info(f"Resumed training from step {start_step}")
    return start_step
