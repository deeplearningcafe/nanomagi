import os
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
import copy

# GPT-4 split pattern optimized with a 2-digit limit for small vocabs
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|"
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,2}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|"
    r"\s*[\r\n]|"
    r"\s+(?!\S)|"
    r"\s+"
)


def bytes_to_unicode():
    """Maps bytes to Unicode characters (standard GPT-2/HF mapping)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(x) for x in cs]
    return dict(zip(bs, cs))


class HFBPETokenizer:
    """
    HuggingFace BPE Tokenizer Wrapper compatible with Nanochat/Toy-Diffusion.
    Uses Byte-Level BPE (BBPE) to represent any Unicode character via 256
    core byte tokens, making the `<|unk|>` token entirely obsolete.
    """

    SPECIAL_TOKENS = [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
    ]

    def __init__(self, tokenizer: Tokenizer, bos_token: str = "<|bos|>"):
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.bos_id = self.tokenizer.token_to_id(bos_token)

        self._byte_encoder = bytes_to_unicode()
        self._char_to_byte = {v: k for k, v in self._byte_encoder.items()}

    @classmethod
    def train_from_iterator(cls, iterator, vocab_size):
        """Trains a new BPE tokenizer on a string iterator."""
        model = BPE()
        tokenizer = Tokenizer(model)

        tokenizer.normalizer = normalizers.NFKC()

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False),
            ]
        )

        tokenizer.decoder = decoders.ByteLevel()

        from tokenizers.trainers import BpeTrainer

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator(iterator, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """Loads the tokenizer from a directory."""
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = Tokenizer.from_file(json_path)
        return cls(tokenizer)

    def save(self, tokenizer_dir):
        """Saves the tokenizer to a directory."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(json_path)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        return set(self.SPECIAL_TOKENS)

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.bos_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """Encodes text or list of texts into token IDs."""
        if isinstance(text, str):
            encoded = self.tokenizer.encode(text)
            ids = list(encoded.ids)
            if prepend is not None:
                p_id = (
                    prepend
                    if isinstance(prepend, int)
                    else self.tokenizer.token_to_id(prepend)
                )
                ids.insert(0, p_id)
            if append is not None:
                a_id = (
                    append
                    if isinstance(append, int)
                    else self.tokenizer.token_to_id(append)
                )
                ids.append(a_id)
            return ids
        elif isinstance(text, list):
            encodings = self.tokenizer.encode_batch(text)
            results = []
            p_id = None
            if prepend is not None:
                p_id = (
                    prepend
                    if isinstance(prepend, int)
                    else self.tokenizer.token_to_id(prepend)
                )
            a_id = None
            if append is not None:
                a_id = (
                    append
                    if isinstance(append, int)
                    else self.tokenizer.token_to_id(append)
                )

            for enc in encodings:
                ids = list(enc.ids)
                if p_id is not None:
                    ids.insert(0, p_id)
                if a_id is not None:
                    ids.append(a_id)
                results.append(ids)
            return results
        else:
            raise TypeError("Text must be a string or a list of strings.")

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def decode_single_token_bytes(self, token_id):
        """Recovers the exact original raw bytes of a single token ID."""
        token_str = self.tokenizer.id_to_token(token_id)
        if token_str is None or token_str in self.get_special_tokens():
            return b""

        try:
            byte_list = [self._char_to_byte[c] for c in token_str]
            return bytes(byte_list)
        except KeyError:
            return token_str.encode("utf-8")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single chat conversation.
        Returns:
        - ids: list[int] of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for assistant tokens.
        """
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = conversation["messages"]
        if messages[0]["role"] == "system":
            messages = copy.deepcopy(messages)
            assert messages[1]["role"] == "user", (
                "System message must be followed by user message"
            )
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]

        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} "
                f"but should be {must_be_from}"
            )
            content = message["content"]
            if message["role"] == "user":
                val_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(val_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                val_ids = self.encode(content)
                add_tokens(assistant_start, 0)
                add_tokens(val_ids, 1)
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        if messages[-1]["role"] == "assistant":
            messages.pop()
        ids, _ = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids