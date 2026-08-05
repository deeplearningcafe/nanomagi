import os
import time
import json
import argparse
import logging
from datetime import datetime
import torch
from omegaconf import OmegaConf

from nanomagi.utils import (
    get_model,
    get_tokenizer,
    gpu_setup,
    get_raw_model,
)
from nanomagi.checkpointing import (
    load_from_checkpoint,
    load_checkpoint_config,
)
from nanomagi.logging_utils import Logger
from nanomagi.engine import Engine


def load_model_and_tokenizer(model_path, device):
    # Read the saved configuration dictionary
    config_data = load_checkpoint_config(model_path)

    config = OmegaConf.create(config_data)

    model = get_model(config, device)

    load_from_checkpoint(model_path, model=model)

    model = get_raw_model(model)
    model.eval()

    tokenizer = get_tokenizer(config)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Nanomagi Single Device Inference Suite"
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable conversational chat mode",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="",
        help="Single prompt execution (non-interactive)",
    )
    parser.add_argument(
        "-t",
        "--temp",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty parameter",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("results", "inference", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    Logger.setup_logging(log_dir, "inference")

    device = torch.device(args.device)
    logging.info(f"Setting up GPU configurations for device: {device}")
    _ = gpu_setup(device)

    logging.info(f"Loading checkpoint from: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    engine = Engine(model, tokenizer)

    log_file = os.path.join(log_dir, "inference_log.json")
    logs = []

    def save_logs():
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    if args.prompt:
        user_input = args.prompt
        start_time = time.time()

        if args.chat:
            conversation_tokens = (
                [bos, user_start]
                + tokenizer.encode(user_input)
                + [user_end, assistant_start]
            )
        else:
            conversation_tokens = tokenizer.encode(user_input, prepend=bos)

        response_tokens = []
        print("\nResponse: ", end="", flush=True)
        for token_column in engine.generate(
            conversation_tokens,
            num_samples=1,
            max_tokens=256,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
        ):
            token = token_column[0]
            response_tokens.append(token)
            text = tokenizer.decode([token])
            print(text, end="", flush=True)
        print()

        elapsed = time.time() - start_time
        response_text = tokenizer.decode(response_tokens)

        logs.append(
            {
                "mode": "chat" if args.chat else "base",
                "interactive": False,
                "prompt": user_input,
                "response": response_text,
                "time_seconds": elapsed,
            }
        )
        save_logs()
        logging.info(f"Response logged. Latency: {elapsed:.3f} seconds.")
        return

    # Interactive Mode
    if args.chat:
        print("\n--- Nanomagi Chat Mode (Interactive) ---")
        print("Commands: 'quit' or 'exit' to exit, 'clear' to reset.\n")
        conversation_tokens = [bos]

        while True:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                conversation_tokens = [bos]
                print("Conversation cleared.")
                continue
            if not user_input:
                continue

            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(user_input))
            conversation_tokens.append(user_end)
            conversation_tokens.append(assistant_start)

            start_time = time.time()
            print("\nAssistant: ", end="", flush=True)
            response_tokens = []
            for token_column in engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=512,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            ):
                token = token_column[0]
                response_tokens.append(token)
                text = tokenizer.decode([token])
                print(text, end="", flush=True)
            print()

            elapsed = time.time() - start_time
            response_text = tokenizer.decode(response_tokens)

            if not response_tokens or response_tokens[-1] != assistant_end:
                response_tokens.append(assistant_end)
            conversation_tokens.extend(response_tokens)

            logs.append(
                {
                    "mode": "chat",
                    "interactive": True,
                    "prompt": user_input,
                    "response": response_text,
                    "time_seconds": elapsed,
                }
            )
            save_logs()

    else:
        print("\n--- Nanomagi Text Completion Mode (Interactive) ---")
        print("Commands: 'quit' or 'exit' to exit.\n")

        while True:
            try:
                user_input = input("\nPrompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input:
                continue

            tokens = tokenizer.encode(user_input, prepend=bos)
            start_time = time.time()
            print("\nCompletion: ", end="", flush=True)

            response_tokens = []
            for token_column in engine.generate(
                tokens,
                num_samples=1,
                max_tokens=256,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            ):
                token = token_column[0]
                response_tokens.append(token)
                text = tokenizer.decode([token])
                print(text, end="", flush=True)
            print()

            elapsed = time.time() - start_time
            response_text = tokenizer.decode(response_tokens)

            logs.append(
                {
                    "mode": "base",
                    "interactive": True,
                    "prompt": user_input,
                    "response": response_text,
                    "time_seconds": elapsed,
                }
            )
            save_logs()


if __name__ == "__main__":
    main()