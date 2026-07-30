import os
import random
import logging
import torch
from collections import Counter
from datasets import load_dataset
from nanomagi.loss_eval import compute_perplexity
from nanomagi.utils import get_raw_model

os.environ["HF_DATASETS_CACHE"] = "data/eval"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def char_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Computes character-level F1 score.
    """
    prediction_chars = list(prediction)
    ground_truth_chars = list(ground_truth)
    common = Counter(prediction_chars) & Counter(ground_truth_chars)
    num_same = sum(common.values())

    if len(prediction_chars) == 0 or len(ground_truth_chars) == 0:
        return float(prediction_chars == ground_truth_chars)

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_chars)
    recall = 1.0 * num_same / len(ground_truth_chars)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_fewshot_examples(dataset, exclude_idx, num_fewshot, seed=42):
    """
    Deterministically samples in-context examples from the dataset split,
    guaranteeing that the current test index is excluded.
    """
    if num_fewshot <= 0:
        return []
    rng = random.Random(seed + exclude_idx)
    available_indices = [i for i in range(len(dataset)) if i != exclude_idx]
    fewshot_indices = rng.sample(
        available_indices, min(num_fewshot, len(available_indices))
    )
    return [dataset[i] for i in fewshot_indices]


def get_choice_ll(model, tokenizer, device, context_tokens, choice_str):
    """
    Calculates the exact token log-likelihood of a choice string given
    the prefix context.
    """
    choice_tokens = tokenizer.encode(choice_str, prepend=None)
    full_tokens = context_tokens + choice_tokens

    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids)

    # We evaluate logits predicting the choice tokens starting from context end
    start_idx = len(context_tokens) - 1
    end_idx = len(full_tokens) - 1

    target_logits = logits[0, start_idx:end_idx, :]
    target_ids = input_ids[0, len(context_tokens) :]

    log_probs = torch.log_softmax(target_logits, dim=-1)
    gathered = log_probs[torch.arange(len(target_ids)), target_ids]
    return gathered.sum().item()


def build_jcommonsenseqa_prompt(item, fewshot_items):
    """Formats multiple-choice items into few-shot context."""
    prompt = ""
    for fs in fewshot_items:
        label = fs["label"]
        correct_choice = fs[f"choice{label}"]
        prompt += f"質問: {fs['question']}\n回答: {correct_choice}\n\n"
    prompt += f"質問: {item['question']}\n回答: "
    return prompt


def evaluate_jcommonsenseqa(
    model, tokenizer, device, num_samples=100, num_fewshot=4, seed=42
):
    """
    Computes JCommonsenseQA validation split accuracy.
    Uses log-likelihood of the choice options directly.
    """
    try:
        ds = load_dataset(
            "sbintuitions/JCommonsenseQA",
            split="validation",
            cache_dir="data/eval",
        )
    except Exception as e:
        logger.warning(f"Failed to load JCommonsenseQA: {e}")
        return 0.0

    correct = 0
    total = min(num_samples, len(ds))
    bos_id = tokenizer.get_bos_token_id()

    for idx in range(total):
        item = ds[idx]
        fewshot_items = get_fewshot_examples(ds, idx, num_fewshot, seed=seed)
        prompt = build_jcommonsenseqa_prompt(item, fewshot_items)
        context_tokens = tokenizer.encode(prompt, prepend=bos_id)

        if len(context_tokens) > 2048:
            context_tokens = context_tokens[-2048:]

        choices = [
            item["choice0"],
            item["choice1"],
            item["choice2"],
            item["choice3"],
            item["choice4"],
        ]

        best_choice_idx = -1
        best_ll = -float("inf")

        for i, choice in enumerate(choices):
            ll = get_choice_ll(model, tokenizer, device, context_tokens, choice)
            if ll > best_ll:
                best_ll = ll
                best_choice_idx = i

        if best_choice_idx == item["label"]:
            correct += 1

    return correct / max(1, total)


def build_jmmlu_prompt(item, fewshot_items):
    """Formats JMMLU items into a few-shot context."""
    prompt = ""
    for fs in fewshot_items:
        label = fs["label"]
        correct_choice = fs[f"choice{label}"]
        prompt += f"質問: {fs['question']}\n回答: {correct_choice}\n\n"
    prompt += f"質問: {item['question']}\n回答: "
    return prompt


def evaluate_jmmlu(model, tokenizer, device, num_samples=100, num_fewshot=4, seed=42):
    """
    Computes JMMLU test split accuracy.
    Uses log-likelihood of choice options.
    """
    try:
        ds = load_dataset(
            "zenless-lab/jmmlu",
            split="test",
            cache_dir="data/eval",
        )
    except Exception as e:
        logger.warning(f"Failed to load JMMLU: {e}")
        return 0.0

    correct = 0
    total = min(num_samples, len(ds))
    bos_id = tokenizer.get_bos_token_id()

    for idx in range(total):
        item = ds[idx]
        fewshot_items = get_fewshot_examples(ds, idx, num_fewshot, seed=seed)
        prompt = build_jmmlu_prompt(item, fewshot_items)
        context_tokens = tokenizer.encode(prompt, prepend=bos_id)

        if len(context_tokens) > 2048:
            context_tokens = context_tokens[-2048:]

        choices = [
            item["choice0"],
            item["choice1"],
            item["choice2"],
            item["choice3"],
        ]

        best_choice_idx = -1
        best_ll = -float("inf")

        for i, choice in enumerate(choices):
            ll = get_choice_ll(model, tokenizer, device, context_tokens, choice)
            if ll > best_ll:
                best_ll = ll
                best_choice_idx = i

        if best_choice_idx == item["label"]:
            correct += 1

    return correct / max(1, total)


def build_jsquad_prompt(item, fewshot_items):
    """Formats JSQuAD context and question pairs into few-shot structures."""
    prompt = ""
    for fs in fewshot_items:
        gold_text = ""
        if fs["answers"]["text"]:
            gold_text = fs["answers"]["text"][0]
        prompt += (
            f"文脈: {fs['context']}\n質問: {fs['question']}\n回答: {gold_text}\n\n"
        )
    prompt += f"文脈: {item['context']}\n質問: {item['question']}\n回答: "
    return prompt


def evaluate_jsquad(model, tokenizer, device, num_samples=100, num_fewshot=4, seed=42):
    """
    Computes JSQuAD validation split exact match (EM) and character F1.
    Uses greedy decoding starting from prompt context.
    """
    try:
        ds = load_dataset(
            "sbintuitions/JSQuAD",
            split="validation",
            cache_dir="data/eval",
        )
    except Exception as e:
        logger.warning(f"Failed to load JSQuAD: {e}")
        return 0.0, 0.0

    total = min(num_samples, len(ds))
    bos_id = tokenizer.get_bos_token_id()

    total_f1 = 0.0
    total_em = 0.0

    for idx in range(total):
        item = ds[idx]
        fewshot_items = get_fewshot_examples(ds, idx, num_fewshot, seed=seed)
        prompt = build_jsquad_prompt(item, fewshot_items)
        tokens = tokenizer.encode(prompt, prepend=bos_id)

        if len(tokens) > 2048:
            tokens = tokens[-2048:]

        generated_ids = list(model.generate(tokens, max_tokens=32, temperature=0.0))
        prediction = tokenizer.decode(generated_ids).strip()
        if "\n" in prediction:
            prediction = prediction.split("\n")[0]
        prediction = prediction.strip()

        gold_answers = item["answers"]["text"]

        best_f1 = 0.0
        best_em = 0.0
        for gold in gold_answers:
            gold = gold.strip()
            f1 = char_f1_score(prediction, gold)
            em = 1.0 if prediction == gold else 0.0
            best_f1 = max(best_f1, f1)
            best_em = max(best_em, em)

        total_f1 += best_f1
        total_em += best_em

    return total_em / max(1, total), total_f1 / max(1, total)


def build_niilc_prompt(item, fewshot_items):
    """Formats NIILC-QA items into few-shot structures."""
    prompt = ""
    for fs in fewshot_items:
        q = fs.get("text") or fs.get("question")
        ans = ""
        if fs.get("answers"):
            ans = fs.get("answers")[0]
        prompt += f"質問: {q}\n回答: {ans}\n\n"

    q_target = item.get("text") or item.get("question")
    prompt += f"質問: {q_target}\n回答: "
    return prompt


def evaluate_niilc_qa(
    model, tokenizer, device, num_samples=100, num_fewshot=4, seed=42
):
    """
    Computes NIILC-QA (v1.2) dev split exact match (EM) and character F1.
    Uses greedy decoding starting from prompt context.
    """
    try:
        ds = load_dataset(
            "sbintuitions/niilc-qa",
            name="v1.2",
            split="dev",
            cache_dir="data/eval",
        )
    except Exception as e:
        try:
            ds = load_dataset(
                "sbintuitions/niilc-qa",
                name="v1.2",
                split="test",
                cache_dir="data/eval",
            )
        except Exception as e2:
            logger.warning(f"Failed to load NIILC-QA: {e2}")
            return 0.0, 0.0

    total = min(num_samples, len(ds))
    bos_id = tokenizer.get_bos_token_id()

    total_f1 = 0.0
    total_em = 0.0

    for idx in range(total):
        item = ds[idx]
        question = item.get("text") or item.get("question")
        if not question:
            continue

        fewshot_items = get_fewshot_examples(ds, idx, num_fewshot, seed=seed)
        prompt = build_niilc_prompt(item, fewshot_items)
        tokens = tokenizer.encode(prompt, prepend=bos_id)

        if len(tokens) > 2048:
            tokens = tokens[-2048:]

        generated_ids = list(model.generate(tokens, max_tokens=32, temperature=0.0))
        prediction = tokenizer.decode(generated_ids).strip()
        if "\n" in prediction:
            prediction = prediction.split("\n")[0]
        prediction = prediction.strip()

        gold_answers = item.get("answers") or []

        best_f1 = 0.0
        best_em = 0.0
        for gold in gold_answers:
            gold = gold.strip()
            f1 = char_f1_score(prediction, gold)
            em = 1.0 if prediction == gold else 0.0
            best_f1 = max(best_f1, f1)
            best_em = max(best_em, em)

        total_f1 += best_f1
        total_em += best_em

    return total_em / max(1, total), total_f1 / max(1, total)


def run_unified_evaluation(
    model,
    tokenizer,
    device,
    val_path=None,
    num_samples=100,
    num_fewshot=4,
    seed=42,
):
    """
    Synchronously runs the evaluations on all 4 Japanese benchmarks
    plus local holdout validation PPL using N-shot prompting.
    """
    raw_model = get_raw_model(model)
    raw_model.eval()

    results = {}

    logger.info(f"Evaluating JCommonsenseQA ({num_fewshot}-shot)...")
    jc_acc = evaluate_jcommonsenseqa(
        raw_model,
        tokenizer,
        device,
        num_samples,
        num_fewshot,
        seed=seed,
    )
    results["eval/jcommonsenseqa_acc"] = jc_acc

    logger.info(f"Evaluating JMMLU ({num_fewshot}-shot)...")
    jmmlu_acc = evaluate_jmmlu(
        raw_model,
        tokenizer,
        device,
        num_samples,
        num_fewshot,
        seed=seed,
    )
    results["eval/jmmlu_acc"] = jmmlu_acc

    logger.info(f"Evaluating JSQuAD ({num_fewshot}-shot)...")
    js_em, js_f1 = evaluate_jsquad(
        raw_model,
        tokenizer,
        device,
        num_samples,
        num_fewshot,
        seed=seed,
    )
    results["eval/jsquad_em"] = js_em
    results["eval/jsquad_f1"] = js_f1

    logger.info(f"Evaluating NIILC-QA ({num_fewshot}-shot)...")
    ni_em, ni_f1 = evaluate_niilc_qa(
        raw_model,
        tokenizer,
        device,
        num_samples,
        num_fewshot,
        seed=seed,
    )
    results["eval/niilc_em"] = ni_em
    results["eval/niilc_f1"] = ni_f1

    if val_path and os.path.exists(val_path):
        logger.info("Evaluating validation perplexity...")
        try:
            ppl, val_loss = compute_perplexity(raw_model, tokenizer, val_path, device)
            results["val/perplexity"] = ppl
            results["val/loss"] = val_loss
        except Exception as e:
            logger.warning(f"Failed computing PPL: {e}")

    raw_model.train()
    return results
