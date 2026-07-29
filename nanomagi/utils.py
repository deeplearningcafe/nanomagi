import os
import torch
import torch.distributed as dist
from nanomagi.gpt import GPT, GPTConfig
from nanomagi.tokenizer import HFBPETokenizer


def get_model(config, device, file_path=None):
    """Builds the GPT model from config and loads state dict if specified."""
    # Convert OmegaConf config block to dict and unpack as kwargs
    gpt_kwargs = dict(config.gpt) if hasattr(config, "gpt") else {}
    gpt_kwargs["use_checkpointing"] = config.training.get(
        "use_gradient_checkpointing", False
    )
    gpt_config = GPTConfig(**gpt_kwargs)
    model = GPT(gpt_config)

    if file_path and os.path.exists(file_path):
        state_dict = torch.load(file_path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v
        model.load_state_dict(clean_state_dict)

    model.to(device)
    return model


def get_tokenizer(config):
    """Loads the HFBPETokenizer from the directory specified in config."""
    tokenizer_dir = config.tokenizer.dir
    return HFBPETokenizer.from_directory(tokenizer_dir)


def gpu_setup(device: str = "cuda"):
    autocast_dtype = torch.float32
    device_type = (
        device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    )
    if device_type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        autocast_dtype = torch.bfloat16
        if capability[0] < 8:
            print(
                f"Warning: bfloat16 specified but GPU capability "
                f"({capability[0]}.{capability[1]}) may not fully support it. "
                f"Consider float16 or float32."
            )
            autocast_dtype = torch.float32

        if capability[0] >= 7 and capability[0] < 8:
            autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
            print("Using high precision for float32 matmul (tensor cores).")
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("Using half precision for float32 matmul (tensor cores).")
        else:
            print(
                "Tensor cores for float32 matmul not optimally supported or GPU is older."
            )
    elif device_type == "xpu":
        print(f"Using xpu device with torch version {torch.__version__}")
        autocast_dtype = torch.bfloat16
    return autocast_dtype


def get_dist_info():
    """Returns DDP status, global rank, local rank, and world size."""
    is_initialized = dist.is_initialized()
    ddp_active = is_initialized and dist.get_world_size() > 1
    rank = dist.get_rank() if is_initialized else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if is_initialized else 1
    return ddp_active, rank, local_rank, world_size


def get_raw_model(model):
    """
    Unwraps model from DDP or compiled wrappers to avoid dynamic-shape
    recompilation overhead during generation.
    """
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model
