import os
import math
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


def evaluate_jcommonsenseqa(model, tokenizer, device, num_samples=100):
    """
    Computes JCommonsenseQA accuracy on the validation split using
    choice-level log-likelihood scoring.
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
        prompt = f"質問: {item['question']}\n回答: "
        context_tokens = tokenizer.encode(prompt, prepend=bos_id)

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


def evaluate_jmmlu(model, tokenizer, device, num_samples=100):
    """
    Computes JMMLU accuracy on the test split using choice-level
    log-likelihood scoring.
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
        prompt = f"質問: {item['question']}\n回答: "
        context_tokens = tokenizer.encode(prompt, prepend=bos_id)

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


def evaluate_jsquad(model, tokenizer, device, num_samples=100):
    """
    Computes JSQuAD EM and Character F1 scores using greedy decoding.
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
        prompt = f"文脈: {item['context']}\n質問: {item['question']}\n回答: "
        tokens = tokenizer.encode(prompt, prepend=bos_id)

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


def evaluate_niilc_qa(model, tokenizer, device, num_samples=100):
    """
    Computes NIILC-QA (v1.2) EM and F1 scores using greedy decoding
    on the dev/validation split.
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
                split="validation",
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

        prompt = f"質問: {question}\n回答: "
        tokens = tokenizer.encode(prompt, prepend=bos_id)

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


def run_unified_evaluation(model, tokenizer, device, val_path=None, num_samples=100):
    """
    Synchronously runs all four Japanese benchmarks and computes
    holdout loss and perplexity.
    """
    raw_model = get_raw_model(model)
    raw_model.eval()

    results = {}

    logger.info("Evaluating JCommonsenseQA...")
    jc_acc = evaluate_jcommonsenseqa(raw_model, tokenizer, device, num_samples)
    results["eval/jcommonsenseqa_acc"] = jc_acc

    logger.info("Evaluating JMMLU...")
    jmmlu_acc = evaluate_jmmlu(raw_model, tokenizer, device, num_samples)
    results["eval/jmmlu_acc"] = jmmlu_acc

    logger.info("Evaluating JSQuAD...")
    js_em, js_f1 = evaluate_jsquad(raw_model, tokenizer, device, num_samples)
    results["eval/jsquad_em"] = js_em
    results["eval/jsquad_f1"] = js_f1

    logger.info("Evaluating NIILC-QA...")
    ni_em, ni_f1 = evaluate_niilc_qa(raw_model, tokenizer, device, num_samples)
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
