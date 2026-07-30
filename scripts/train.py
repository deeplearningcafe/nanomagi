import os
import argparse
from omegaconf import OmegaConf
import logging
from datetime import datetime

from nanomagi.trainer import Trainer
from nanomagi.evaluator import run_unified_evaluation
from nanomagi.logging_utils import Logger


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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{cfg.training.save_dir}/{timestamp}"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"{cfg.experiment.name}",
    )
    logging.info(cfg)

    trainer = Trainer(cfg)

    trainer.fit(timestamp)

    print("Pretraining complete. Running evaluations...")
    eval_results = run_unified_evaluation(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        device=trainer.device,
        val_path=trainer.val_path,
        num_samples=200,  # Evaluate on more samples for final results
        num_fewshot=4,
    )
    for k, v in eval_results.items():
        print(f"Final {k}: {v:.4f}")
        if trainer.use_wandb:
            trainer.wandb.log({f"final/{k}": v})


if __name__ == "__main__":
    main()
