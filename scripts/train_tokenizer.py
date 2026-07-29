import os
import time
import argparse
import torch
import unicodedata
from nanomagi.tokenizer import HFBPETokenizer
from nanomagi.dataset import get_mixed_streaming_dataset

parser = argparse.ArgumentParser(description="Train a streaming BPE tokenizer")
parser.add_argument(
    "--max-chars",
    type=int,
    default=500_000_000,
    help="Max characters to train on (default: 500M)",
)
parser.add_argument(
    "--doc-cap",
    type=int,
    default=10_000,
    help="Max characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=32768,
    help="Vocabulary size (default: 32768)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="tokenizer",
    help="Directory to save the tokenizer",
)
args = parser.parse_args()

print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")


def text_iterator_streaming(max_chars, doc_cap):
    """
    Streams text dynamically from the pretraining mixed dataset.
    Preserves 100% of the disk and memory limitations.
    """
    dataset = get_mixed_streaming_dataset(stage=1, seed=1255)
    nchars = 0

    for doc in dataset:
        text = doc.get("text", "")
        if len(text) > doc_cap:
            text = text[:doc_cap]
        nchars += len(text)
        yield text
        if nchars >= max_chars:
            break


text_iter = text_iterator_streaming(args.max_chars, args.doc_cap)

t0 = time.time()
tokenizer = HFBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
print(f"Training time: {t1 - t0:.2f}s")

tokenizer.save(args.output_dir)

test_text = """こんにちは世界！ これはテストです。
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""


# Apply Unicode NFKC normalization prior to equivalence verification
normalized_test = unicodedata.normalize("NFKC", test_text)
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

assert decoded == normalized_test, (
    f"Decoded {decoded} and normalized test {normalized_test}"
)

# Save token_bytes for bits-per-byte evaluation
vocab_size = tokenizer.get_vocab_size()
special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)
    else:
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)

token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(args.output_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
