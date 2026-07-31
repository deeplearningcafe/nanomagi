import os
import time
import logging
import torch
import numpy as np
import random
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from nanomagi.dataloader import (
    tokenizing_distributed_data_loader_with_state_bos_bestfit as get_dataloader,
)
from nanomagi.utils import (
    get_model,
    get_tokenizer,
    gpu_setup,
    get_dist_info,
    get_raw_model,
)
from nanomagi.optim import create_optim_scheduler
from nanomagi.checkpointing import (
    save_checkpoint,
    load_from_checkpoint,
)
from nanomagi.callbacks import log_generations
from nanomagi.act_grad_checkpointing import (
    patch_unsloth_smart_gradient_checkpointing,
    patch_torch_compile,
    patch_compiled_autograd,
    CPUGradientAccumulator,
)
from nanomagi.dataset import build_static_validation_set
from nanomagi.evaluator import run_unified_evaluation

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.setup_device()

        self.use_wandb = self.config.logging.get("use_wandb", False)
        if self.use_wandb and self.global_rank == 0:
            import wandb

            self.wandb = wandb
            self.wandb.init(
                project=self.config.logging.project,
                name=self.config.logging.get("run_name", None),
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        self.tokenizer = get_tokenizer(self.config)
        self.model = get_model(self.config, self.device)

        # Compute training horizon, flops, and tokens per step
        self.batch_size = self.config.training.get("batch_size", 16)
        self.max_seq_len = self.config.gpt.get("max_position_embeddings", 2048)
        self.grad_clip = self.config.training.get("grad_clip", 1.0)
        self.grad_accum_steps = self.config.training.get(
            "gradient_accumulation_steps", 1
        )
        self.tokens_per_step = (
            self.batch_size * self.max_seq_len * self.grad_accum_steps * self.world_size
        )

        # Estimate FLOPs per token and non-embedding params
        self.num_flops_per_token = self.model.estimate_flops()
        self.non_embed_params = self.model.non_embed_params()

        num_iterations_cfg = self.config.training.get("num_iterations", -1)
        target_flops = self.config.training.get("target_flops", -1.0)
        target_ratio = self.config.training.get("target_param_data_ratio", -1)

        if num_iterations_cfg > 0:
            self.num_iterations = num_iterations_cfg
        elif target_flops > 0:
            self.num_iterations = int(
                target_flops / (self.num_flops_per_token * self.tokens_per_step)
            )
        elif target_ratio > 0:
            target_tokens = int(target_ratio * self.non_embed_params)
            self.num_iterations = int(target_tokens / self.tokens_per_step)
        else:
            raise ValueError("No valid training horizon specified in config.")

        if self.global_rank == 0:
            logger.info(
                f"Model Non-Embedding Params: "
                f"{self.non_embed_params / 1e6:.2f}M | "
                f"Embedding Params: "
                f"{self.model.embed_params() / 1e6:.2f}M"
            )
            logger.info(
                f"FLOPs per token: {self.num_flops_per_token:.2e} | "
                f"Tokens per step: {self.tokens_per_step:,}"
            )
            logger.info(f"Total training iterations: {self.num_iterations:,}")

        self.optimizer, self.scheduler = create_optim_scheduler(
            self.model,
            total_steps=self.num_iterations,
            conf=self.config,
        )

        self.start_step = 0
        resume_dir = self.config.training.get("resume_from_checkpoint", None)
        if resume_dir:
            self.start_step = load_from_checkpoint(
                checkpoint_dir=resume_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.autocast_dtype == torch.float16
            else None
        )

        self.prompts = self._load_sample_configs()

        self.eval_interval = self.config.training.get("eval_interval", 100)
        self.sample_interval = self.config.training.get("sample_interval", 50)
        # it should be the same with the seed so no need to timestamp
        self.val_path = self.config.training.get("val_path", "data/eval.json")
        dirname = os.path.dirname(self.val_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        self.num_val_samples = self.config.data.get("num_val_samples", 10000)
        self.save_interval = self.config.training.get("save_interval", 1000)
        self.output_dir = self.config.training.get("save_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

        if hasattr(torch, "compile") and self.config.training.get(
            "compile_model", True
        ):
            logger.info("Compiling model for faster training...")
            patch_torch_compile()
            patch_compiled_autograd()
            self.model = torch.compile(self.model)

        if self.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        self.grad_offloader = None
        if self.grad_accum_steps > 1:
            self.grad_offloader = CPUGradientAccumulator(self.model)

    def _load_sample_configs(self):
        prompts = []
        config_file = self.config.sampling.get("sample_file", None)
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                prompts = f.read().splitlines()
        if len(prompts) == 0:
            prompts = [
                "昔々、あるところに",
                "吾輩は猫である。名前はまだ無い。",
                "富士山は日本で最も高い山です。",
            ]
        print(f"Loaded {len(prompts)} prompts.")
        return prompts

    def setup_device(self):
        ddp_active, rank, l_rank, world_size = get_dist_info()
        self.is_ddp = ddp_active
        self.global_rank = rank
        self.local_rank = l_rank
        self.world_size = world_size

        if self.is_ddp and not dist.is_initialized():
            print(
                f"Initializing DDP: Rank {self.global_rank}/"
                f"{self.world_size}, Local Rank {self.local_rank}"
            )
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        self.seed = self.config.training.get("seed", 42) + self.global_rank
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.global_rank == 0:
            dtype_str = self.config.training.get("dtype", "bf16")
            print(f"Training on {self.world_size} GPUs. Precision: {dtype_str}")

        self.autocast_dtype = gpu_setup(self.device)
        print(f"Using {self.autocast_dtype} for autocast")
        patch_unsloth_smart_gradient_checkpointing(self.autocast_dtype)

    def train_step(self, inputs, targets):
        """Executes a single forward and backward pass."""
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        device_type = "cuda" if "cuda" in str(self.device) else str(self.device)
        with torch.amp.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        ):
            loss = self.model(inputs, targets=targets)

        scaled_loss = loss / self.grad_accum_steps
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.detach()

    def fit(self, timestamp=None):
        """Runs the pretraining loop using step/iteration-based training."""
        logger.info("Initializing Autoregressive LLM pretraining...")
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(save_path, exist_ok=True)

        step = self.start_step
        stage = self.config.data.get("stage", 1)
        train_loader = get_dataloader(
            tokenizer=self.tokenizer,
            B=self.batch_size,
            T=self.max_seq_len,
            split="train",
            stage=stage,
            seed=self.seed,
            num_val_samples=self.num_val_samples,
            device=self.device,
        )
        # build val dataset
        build_static_validation_set(
            output_path=self.val_path, num_samples=self.num_val_samples, seed=self.seed
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            total=self.num_iterations,
            initial=step,
            desc="Training Steps",
            disable=self.global_rank != 0,
        )

        while step < self.num_iterations:
            accum_loss = torch.tensor([0.0], device=self.device)
            for micro_step in range(self.grad_accum_steps):
                inputs, targets, state_dict = next(train_loader)
                loss_val = self.train_step(inputs, targets)
                accum_loss += loss_val

            if self.grad_offloader is not None:
                norm_val = self.grad_offloader.finalize_and_step(
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    max_norm=self.grad_clip,
                )
            else:
                norm_val = 0.0
                if self.scaler:
                    if self.grad_clip > 0.0:
                        self.scaler.unscale_(self.optimizer)
                        norm_val = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0.0:
                        norm_val = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            step += 1
            pbar.update(1)

            avg_loss = accum_loss / self.grad_accum_steps
            if self.global_rank == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if self.use_wandb:
                    self.wandb.log(
                        {
                            "train/loss": avg_loss.item(),
                            "train/lr": lr,
                            "train/grad_norm": norm_val.item(),
                            "step": step,
                        }
                    )
                pbar.set_postfix({"loss": f"{avg_loss.item():.4f}", "lr": f"{lr:.2e}"})

            if self.sample_interval > 0 and step % self.sample_interval == 0:
                if self.global_rank == 0:
                    logger.info(f"Generating samples at step {step}")

                    log_generations(
                        model=get_raw_model(self.model),
                        tokenizer=self.tokenizer,
                        step=step,
                        device=self.device,
                        prompts=self.prompts,
                        output_dir=self.output_dir,
                        temperature=1.0,
                    )
                if self.is_ddp:
                    dist.barrier()

            # Benchmark evaluation
            if self.eval_interval > 0 and step % self.eval_interval == 0:
                if self.global_rank == 0:
                    logger.info(f"Running benchmark evaluation at step {step}")
                    try:
                        eval_results = run_unified_evaluation(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=self.device,
                            val_path=self.val_path,
                            num_samples=100,
                            num_fewshot=4,
                            seed=self.seed,
                        )
                        if self.use_wandb:
                            eval_results["step"] = step
                            self.wandb.log(eval_results)
                        for k, v in eval_results.items():
                            logger.info(f"{k}: {v:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed unified evaluation: {e}")
                if self.is_ddp:
                    dist.barrier()

            if self.save_interval > 0 and step % self.save_interval == 0:
                if self.global_rank == 0:
                    logger.info(f"Saving checkpoint at step {step}")
                    save_checkpoint(
                        output_dir=save_path,
                        step=step,
                        model=self.model.module if self.is_ddp else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        config=OmegaConf.to_container(self.config, resolve=True),
                    )
                if self.is_ddp:
                    dist.barrier()

        pbar.close()
        if self.use_wandb and self.global_rank == 0:
            self.wandb.finish()
