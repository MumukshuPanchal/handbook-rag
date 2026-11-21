"""Count tokens the way the embedding model will.

Chunk sizes are measured in the embedding model's own tokens. A model reads a fixed
maximum number of tokens and silently ignores the rest, so a chunk measured in
characters or words could be cut off without anyone noticing.
"""

from collections.abc import Sequence
from functools import lru_cache

from tokenizers import Tokenizer

from handbook_rag.ingest.chunker import TokenCounter


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(model_name)
    # Count the whole text. The tokenizer file ships with the model's limit switched on.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def token_counter(model_name: str) -> TokenCounter:
    """Return a function that counts tokens with the tokenizer of ``model_name``."""
    tokenizer = _load_tokenizer(model_name)

    def count(texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]

    return count
