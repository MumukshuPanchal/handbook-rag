import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from handbook_rag.ingest import tokens


@pytest.fixture
def tiny_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the downloaded tokenizer with a local one, so the test needs no network."""
    vocabulary = {"[UNK]": 0, "parental": 1, "leave": 2, "is": 3, "paid": 4}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    # Mimic a model file that ships with truncation switched on.
    tokenizer.enable_truncation(max_length=2)
    monkeypatch.setattr(Tokenizer, "from_pretrained", lambda _name: tokenizer)
    tokens._load_tokenizer.cache_clear()


@pytest.mark.usefixtures("tiny_tokenizer")
def test_counts_every_token_even_past_the_model_limit() -> None:
    count = tokens.token_counter("any-model")

    assert count(["parental leave is paid", "leave", ""]) == [4, 1, 0]


@pytest.mark.usefixtures("tiny_tokenizer")
def test_empty_input_gives_empty_output() -> None:
    assert tokens.token_counter("any-model")([]) == []
