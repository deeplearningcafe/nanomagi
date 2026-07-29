import os
import argparse
from omegaconf import OmegaConf

from nanomagi.trainer import Trainer
from nanomagi.evaluator import run_unified_evaluation


def main():
    parser = argparse.ArgumentParser(description="Pretrain Autoregressive Japanese LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML training configuration",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")

    cfg = OmegaConf.load(args.config)

    trainer = Trainer(cfg)

    trainer.fit()

    print("Pretraining complete. Running evaluations...")
    eval_results = run_unified_evaluation(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        device=trainer.device,
        val_path=trainer.val_path,
        num_samples=200,  # Evaluate on more samples for final results
    )
    for k, v in eval_results.items():
        print(f"Final {k}: {v:.4f}")
        if trainer.use_wandb:
            trainer.wandb.log({f"final/{k}": v})


if __name__ == "__main__":
    main()
