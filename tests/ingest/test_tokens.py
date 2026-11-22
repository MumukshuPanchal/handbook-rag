import socket
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from handbook_rag.ingest import tokens

type Call = tuple[str, str, dict[str, Any]]


@pytest.fixture
def tokenizer_file(tmp_path: Path) -> str:
    """A small tokenizer saved the way a model's tokenizer is: as one JSON file."""
    vocabulary = {"[UNK]": 0, "parental": 1, "leave": 2, "is": 3, "paid": 4}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    # Mimic a model file that ships with truncation and padding switched on. Left on,
    # either would make every count wrong: cut to two, or filled up to six.
    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=6)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    tokens._load_tokenizer.cache_clear()
    return str(path)


def fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    *,
    on_disk: bool = True,
    error: Exception | None = None,
) -> list[Call]:
    """Replace the download function, and return the list its calls are noted in."""
    calls: list[Call] = []

    def download(model_name: str, filename: str, **options: Any) -> str:
        calls.append((model_name, filename, options))
        if error is not None:
            raise error
        if options.get("local_files_only") and not on_disk:
            raise FileNotFoundError("not in the cache")
        return path

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return calls


def test_counts_every_token_even_past_the_model_limit(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    fake_hub(monkeypatch, tokenizer_file)

    count = tokens.token_counter("any-model")

    assert count(["parental leave is paid", "leave", ""]) == [4, 1, 0]


def test_empty_input_gives_empty_output(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    fake_hub(monkeypatch, tokenizer_file)

    assert tokens.token_counter("any-model")([]) == []


def test_a_tokenizer_on_disk_is_read_without_asking_the_hub(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    calls = fake_hub(monkeypatch, tokenizer_file)

    tokens.token_counter("some/model")

    assert calls == [("some/model", "tokenizer.json", {"local_files_only": True})]


def test_a_tokenizer_that_is_not_on_disk_is_downloaded(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    calls = fake_hub(monkeypatch, tokenizer_file, on_disk=False)

    count = tokens.token_counter("some/model")

    assert count(["parental leave"]) == [2]
    # The second call leaves the option out, so the library asks the Hub as it does by default.
    assert calls == [
        ("some/model", "tokenizer.json", {"local_files_only": True}),
        ("some/model", "tokenizer.json", {}),
    ]


def test_an_error_that_is_not_about_a_missing_file_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    calls = fake_hub(monkeypatch, tokenizer_file, error=ValueError("not a model name"))

    with pytest.raises(ValueError, match="not a model name"):
        tokens.token_counter("some/model")

    assert len(calls) == 1


def test_the_tokenizer_is_loaded_once_per_model(
    monkeypatch: pytest.MonkeyPatch, tokenizer_file: str
) -> None:
    calls = fake_hub(monkeypatch, tokenizer_file)

    tokens.token_counter("some/model")
    tokens.token_counter("some/model")

    assert len(calls) == 1


@pytest.mark.slow
def test_a_tokenizer_on_disk_is_loaded_without_looking_up_any_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "BAAI/bge-small-en-v1.5"
    # Make sure the file is on disk. This is the one step that may download it.
    tokens._tokenizer_file(model)
    tokens._load_tokenizer.cache_clear()
    looked_up: list[str] = []
    real_lookup = socket.getaddrinfo

    def recording_lookup(host: Any, *args: Any, **options: Any) -> Any:
        looked_up.append(str(host))
        return real_lookup(host, *args, **options)

    monkeypatch.setattr(socket, "getaddrinfo", recording_lookup)

    count = tokens.token_counter(model)

    assert count(["Parental leave is sixteen weeks."]) == [6]
    # A request to the Hub would have to look up its address first.
    assert looked_up == []
    tokens._load_tokenizer.cache_clear()
