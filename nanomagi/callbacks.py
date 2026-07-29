import torch
import logging
import os

logger = logging.getLogger(__name__)


@torch.no_grad()
def log_generations(
    model, tokenizer, step, device, prompts, output_dir="results", max_gen_tokens=64
):
    """
    Deterministic training callback using greedy decoding (temperature=0.0)
    to trace qualitative generative progression over steps.
    Saves outputs to a local Markdown tracker.
    """
    model.eval()
    log_data = []

    bos_id = tokenizer.get_bos_token_id()

    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=bos_id)

        generated_ids = list(
            model.generate(
                tokens,
                max_tokens=max_gen_tokens,
                temperature=0.0,
            )
        )
        decoded_text = tokenizer.decode(generated_ids)
        log_data.append(
            f"**Prompt:** {prompt}\n\n**Generation:** {decoded_text}\n\n---"
        )

    log_content = f"### Step {step} Generations\n\n" + "\n".join(log_data)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/generation_logs.md", "a", encoding="utf-8") as f:
        f.write(log_content + "\n\n")

    logger.info(f"Logged inline text completions for step {step}.")
    model.train()
