import torch
import logging
import os

logger = logging.getLogger(__name__)


@torch.no_grad()
def log_generations(
    model,
    tokenizer,
    step,
    device,
    prompts,
    output_dir="results",
    max_gen_tokens=64,
    temperature=1.0,
    is_chat=False,
):
    """
    Training callback to trace qualitative generative progression over steps.
    Saves outputs to a local Markdown tracker.
    """
    model.eval()
    log_data = []

    bos_id = tokenizer.get_bos_token_id()

    for prompt in prompts:
        if is_chat:
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "あなたは親切なAIアシスタントです。"
                    },
                    {"role": "user", "content": prompt}
                ]
            }
            tokens = tokenizer.render_for_completion(conversation)
            eos_id = tokenizer.encode_special("<|assistant_end|>")
            generated_ids = list(
                model.generate_chat(
                    tokens,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
                    eos_token_id=eos_id,
                )
            )
        else:
            tokens = tokenizer.encode(prompt, prepend=bos_id)
            generated_ids = list(
                model.generate(
                    tokens,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
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
