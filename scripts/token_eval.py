import os
import argparse
import unicodedata
import codecs
from transformers import AutoTokenizer
from nanomagi.tokenizer import HFBPETokenizer

ja_lit = (
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶して"
    "いる。吾輩はここで始めて人間というものを見た。しかもあとで聞くと"
    "それは書生という人間中で一番獰悪な種族であったそうだ。この書生と"
    "いうのは時々我々を捕えて煮て食うという話である。"
)

ja_tech = (
    "最新の大規模言語モデル（LLM）は、日本語の理解と生成において優れた"
    "性能を発揮します。トークナイザーの設計や語彙サイズの選択は、モデルの"
    "学習効率や推論時の圧縮率に直接的な影響を与えます。本モデルでは、"
    "Byte-Level BPEを採用し、未知語（UNK）の発生を完全に防いでいます。"
)

en_tech = (
    "Autoregressive language models predict the next token based on the "
    "preceding context. Utilizing Byte-Level BPE (BBPE) eliminates "
    "out-of-vocabulary terms by representing arbitrary Unicode text as raw "
    "bytes, ensuring complete structural robustness across any domain."
)

bilingual = (
    "AIエンジニアとして LLM の開発における最大の課題は tokenization の"
    "最適化です。Japanese and English text require robust parsing strategies "
    "to maximize pretraining throughput and reduce vocabulary footprint."
)

all_text = [
    ("Japanese Literature", ja_lit),
    ("Japanese Technical", ja_tech),
    ("English Technical", en_tech),
    ("Bilingual Mixed", bilingual),
]

parser = argparse.ArgumentParser(description="Evaluate Japanese tokenizer")
parser.add_argument(
    "--tokenizer-dir",
    type=str,
    default="tokenizer",
    help="Path to trained nanomagi tokenizer directory",
)
args = parser.parse_args()

if not os.path.exists(os.path.join(args.tokenizer_dir, "tokenizer.json")):
    print(
        f"Error: Trained tokenizer not found at {args.tokenizer_dir}. "
        "Please run 'python -m nanomagi.scripts.train_tokenizer' first."
    )
    exit(1)

pretrained_models = {
    "sarashina2.2": "sbintuitions/sarashina2.2-0.5b",
    "llm-jp-3": "llm-jp/llm-jp-3-150m",
    "gemma-4": "google/gemma-4-26B-A4B-it",
}

pretrained_cache_dir = "tokenizer/pretrained"
os.makedirs(pretrained_cache_dir, exist_ok=True)

tokenizers = {}
vocab_sizes = {}

ours = HFBPETokenizer.from_directory(args.tokenizer_dir)
tokenizers["Ours (Custom)"] = ours
vocab_sizes["Ours (Custom)"] = ours.get_vocab_size()

for name, model_id in pretrained_models.items():
    print(f"Loading {name} ({model_id}) tokenizer into local cache...")
    try:
        tk = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=pretrained_cache_dir,
            trust_remote_code=True,
        )
        tokenizers[name] = tk
        vocab_sizes[name] = len(tk)
    except Exception as e:
        print(f"Failed to load tokenizer for {model_id}: {e}")

# Validate and compute compression stats
results = {}
for name, tokenizer in tokenizers.items():
    results[name] = {}
    for text_name, raw_text in all_text:
        text = unicodedata.normalize("NFKC", raw_text)

        if hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(text)
        else:
            encoded = tokenizer(text)["input_ids"]

        raw_bytes = text.encode("utf-8")
        num_bytes = len(raw_bytes)
        num_tokens = len(encoded)
        ratio = num_bytes / num_tokens if num_tokens > 0 else 0

        results[name][text_name] = {
            "bytes": num_bytes,
            "tokens": num_tokens,
            "ratio": ratio,
        }

# Table colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

print("\n" + "=" * 50)
print(f"{'Tokenizer Name':<25} | {'Vocabulary Size':<15}")
print("-" * 50)
for name, size in vocab_sizes.items():
    print(f"{name:<25} | {size:<15,}")
print("=" * 50 + "\n")


def visualize_tokenizer_output(text_title, raw_text):
    """Prints token boundaries using alternating color coding."""
    text = unicodedata.normalize("NFKC", raw_text)
    print("\n" + "-" * 80)
    print(f"VISUALIZATION: {text_title}")
    print("-" * 80)

    for tok_name, tokenizer in tokenizers.items():
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer(text)["input_ids"]

        parts = []

        # Initialize incremental decoder to handle multi-token byte boundaries
        decoder = codecs.getincrementaldecoder("utf-8")()

        for token_id in ids:
            if hasattr(tokenizer, "decode_single_token_bytes"):
                b = tokenizer.decode_single_token_bytes(token_id)
                if not b:
                    t_str = tokenizer.tokenizer.id_to_token(token_id) or ""
                    # Feed raw string representation to decoder
                    decoded_chunk = decoder.decode(t_str.encode("utf-8"))
                else:
                    # Incrementally decode byte sequence
                    decoded_chunk = decoder.decode(b)
            else:
                decoded = tokenizer.decode([token_id])
                if not decoded:
                    tok_str = tokenizer.convert_ids_to_tokens(token_id)
                    decoded_chunk = str(tok_str).replace("Ġ", " ").replace(" ", " ")
                else:
                    decoded_chunk = decoded
            parts.append(decoded_chunk)

        remaining = decoder.decode(b"", final=True)
        if remaining:
            parts.append(remaining)

        colors = ["\033[92m", "\033[94m"]  # Alternating Green and Blue
        colored_repr = []
        for i, part in enumerate(parts):
            col = colors[i % 2]
            colored_repr.append(f"{col}{part}{RESET}")

        print(f"\n[{tok_name}]")
        print("".join(colored_repr))
    print("-" * 80)


def print_performance_table(all_text, results):
    """Prints full comparative statistics table."""
    for text_name, _ in all_text:
        print(f"\nCompression Performance: {text_name}")
        print("=" * 80)
        print(
            f"{'Tokenizer':<20} | {'Bytes':<8} | {'Tokens':<8} | "
            f"{'Bytes/Token':<12} | {'Relative Diff':<15}"
        )
        print("-" * 80)

        ours_data = results["Ours (Custom)"][text_name]
        ours_ratio = ours_data["ratio"]

        for name in tokenizers.keys():
            data = results[name][text_name]
            ratio = data["ratio"]

            if ours_ratio > 0:
                rel_diff = ((ratio - ours_ratio) / ours_ratio) * 100
                diff_str = f"{rel_diff:+.2f}%"
            else:
                diff_str = "N/A"

            # Apply color cues
            color = GREEN if ratio >= ours_ratio else BLUE
            print(
                f"{name:<20} | {data['bytes']:<8} | {data['tokens']:<8} | "
                f"{color}{ratio:<12.3f}{RESET} | {diff_str:<15}"
            )
        print("=" * 80)


print_performance_table(all_text, results)

for title, text in all_text:
    visualize_tokenizer_output(title, text)
