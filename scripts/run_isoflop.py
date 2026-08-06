import os
import csv
import time
import math
import argparse
import logging
from datetime import datetime
from omegaconf import OmegaConf

from nanomagi.trainer import Trainer
from nanomagi.logging_utils import Logger


def round_to_multiple(number: int, multiple: int) -> int:
    """Rounds an integer to nearest multiple of a given factor."""
    return ((number + multiple - 1) // multiple) * multiple


def parse_args():
    parser = argparse.ArgumentParser(
        description="IsoFLOP Scaling Laws Experiment Sweeper"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Base configuration YAML file",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1e18,
        help="Target FLOP budget per run (default: 1e18)",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[8, 10, 12, 14],
        help="List of model depths to sweep (default: 8 10 12 14)",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=int,
        default=64,
        help="Aspect ratio (n_embd = depth * aspect_ratio)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/isoflop_results.csv",
        help="Path to save CSV benchmark results",
    )
    return parser.parse_args()


def run_isoflop_sweep():
    args = parse_args()
    base_cfg = OmegaConf.load(args.config)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    csv_exists = os.path.exists(args.output_csv)

    fieldnames = [
        "depth",
        "n_embd",
        "device_batch_size",
        "grad_accum_steps",
        "non_embed_params",
        "total_params",
        "target_flops",
        "num_iterations",
        "tokens_trained",
        "val_loss",
        "val_bpb",
        "val_ppl",
        "train_time_sec",
    ]

    ref_lr = base_cfg.optimizer.get("lr", 6e-4)
    ref_embd = 12 * args.aspect_ratio

    for depth in args.depths:
        n_embd = depth * args.aspect_ratio
        n_head = depth
        n_kv_head = max(1, n_head // 2)
        intermediate_size = round_to_multiple(int(n_embd * (8 / 3)), 128)

        # Scale learning rate inversely with sqrt(n_embd)
        scaled_lr = ref_lr * math.sqrt(ref_embd / n_embd)

        # Adapt device batch size and grad accum based on depth & VRAM
        # Base case (d=12, n_embd=768): device_batch_size=32, grad_accum=4
        if depth >= 28:
            device_batch_size = 8
        elif depth >= 16:
            device_batch_size = 16
        else:
            # Capped at 32 for smaller models (d <= 12)
            device_batch_size = 32

        # Reciprocally adjust gradient accumulation to preserve 128 seq/GPU
        grad_accum_steps = (32 * 4) // device_batch_size

        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.gpt.n_layer = depth
        cfg.gpt.n_embd = n_embd
        cfg.gpt.n_head = n_head
        cfg.gpt.n_kv_head = n_kv_head
        cfg.gpt.intermediate_size = intermediate_size

        cfg.training.batch_size = device_batch_size
        cfg.training.gradient_accumulation_steps = grad_accum_steps

        cfg.optimizer.lr = scaled_lr
        cfg.training.target_flops = args.target_flops
        cfg.training.num_iterations = -1
        cfg.training.target_param_data_ratio = -1
        cfg.training.resume_from_checkpoint = None
        cfg.training.save_interval = -1
        cfg.training.eval_interval = -1
        cfg.training.sample_interval = -1


        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"isoflop_d{depth}_{timestamp}"
        cfg.experiment.name = exp_name

        save_dir = os.path.join(cfg.training.save_dir, exp_name)
        Logger.setup_logging(save_dir=save_dir, logging_name=exp_name)

        logger = logging.getLogger(__name__)
        logger.info(
            f"IsoFLOP run: Depth={depth}, n_embd={n_embd}, "
            f"BatchSize={device_batch_size}, GradAccum={grad_accum_steps}, "
            f"LR={scaled_lr:.2e}, FLOPs={args.target_flops:.1e}"
        )

        t0 = time.time()
        trainer = Trainer(cfg)
        eval_metrics = trainer.fit(timestamp=timestamp)
        t1 = time.time()
        elapsed_sec = t1 - t0

        val_loss = eval_metrics.get("val/loss", float("nan"))
        val_bpb = eval_metrics.get("val/bpb", float("nan"))
        val_ppl = eval_metrics.get("val/perplexity", float("nan"))

        total_tokens = trainer.tokens_per_step * trainer.num_iterations

        row_data = {
            "depth": depth,
            "n_embd": n_embd,
            "device_batch_size": device_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "non_embed_params": trainer.non_embed_params,
            "total_params": sum(p.numel() for p in trainer.model.parameters()),
            "target_flops": args.target_flops,
            "num_iterations": trainer.num_iterations,
            "tokens_trained": total_tokens,
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "val_ppl": val_ppl,
            "train_time_sec": round(elapsed_sec, 2),
        }

        global_rank = int(os.environ.get("RANK", 0))
        if global_rank == 0:
            with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not csv_exists:
                    writer.writeheader()
                    csv_exists = True
                writer.writerow(row_data)

        logger.info(
            f"Done Depth={depth} in {elapsed_sec:.1f}s | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val BPB: {val_bpb:.2f}"
        )


if __name__ == "__main__":
    run_isoflop_sweep()