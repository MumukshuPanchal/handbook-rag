"""Models are read from disk first, and the Hub is asked only for what is missing."""

import importlib.util
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

import pytest

from handbook_rag import local_models
from handbook_rag.local_models import _switched_off, load_local_first, quiet_model_libraries

TORCH_NOTE_LOGGER = "torch.distributed.elastic.multiprocessing.redirects"
NOT_ON_DISK = OSError("couldn't find them in the cached files")

type Call = tuple[str, dict[str, object]]


def recording_loader(
    calls: list[Call], *, on_disk: bool = True, on_hub: bool = True
) -> Callable[..., str]:
    """Stand in for a model class: note each call, and fail where the model is missing."""

    def load(model_name: str, **options: object) -> str:
        calls.append((model_name, options))
        if options.get("local_files_only"):
            if not on_disk:
                raise NOT_ON_DISK
        elif not on_hub:
            raise OSError("the Hub cannot be reached")
        return f"the model {model_name}"

    return load


@pytest.fixture
def no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real switch imports the transformers library, which takes about a second.
    monkeypatch.setattr(local_models, "_no_progress_bars", nullcontext)


@pytest.mark.usefixtures("no_library")
def test_a_model_on_disk_is_loaded_without_asking_the_hub() -> None:
    calls: list[Call] = []

    model = load_local_first(recording_loader(calls), "some/model", device="cpu")

    assert model == "the model some/model"
    assert calls == [("some/model", {"local_files_only": True, "device": "cpu"})]


@pytest.mark.usefixtures("no_library")
def test_a_model_that_is_not_on_disk_is_fetched_from_the_hub() -> None:
    calls: list[Call] = []

    model = load_local_first(recording_loader(calls, on_disk=False), "some/model", device="cpu")

    assert model == "the model some/model"
    # The second call leaves the option out, so the library behaves as it does by default.
    assert calls == [
        ("some/model", {"local_files_only": True, "device": "cpu"}),
        ("some/model", {"device": "cpu"}),
    ]


@pytest.mark.usefixtures("no_library")
def test_when_the_hub_fails_too_its_error_is_the_one_reported() -> None:
    calls: list[Call] = []
    load = recording_loader(calls, on_disk=False, on_hub=False)

    with pytest.raises(OSError, match="the Hub cannot be reached"):
        load_local_first(load, "some/model")

    assert len(calls) == 2


@pytest.mark.usefixtures("no_library")
def test_an_error_that_is_not_about_missing_files_is_not_retried() -> None:
    calls: list[str] = []

    def load(model_name: str, **options: object) -> str:
        calls.append(model_name)
        raise ValueError("this is not an embedding model")

    with pytest.raises(ValueError, match="not an embedding model"):
        load_local_first(load, "some/model")

    assert calls == ["some/model"]


def test_progress_is_hidden_while_reading_from_disk_and_shown_for_a_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def hidden() -> Iterator[None]:
        events.append("bars hidden")
        try:
            yield
        finally:
            events.append("bars back")

    def load(model_name: str, **options: object) -> str:
        if options.get("local_files_only"):
            events.append("read from disk")
            raise NOT_ON_DISK
        events.append("download")
        return "the model"

    monkeypatch.setattr(local_models, "_no_progress_bars", hidden)

    load_local_first(load, "some/model")

    assert events == ["bars hidden", "read from disk", "bars back", "download"]


class Switch:
    """Something that can be on or off, such as a library's progress bars."""

    def __init__(self, *, on: bool) -> None:
        self.on = on

    def is_on(self) -> bool:
        return self.on

    def switch_off(self) -> None:
        self.on = False

    def switch_on(self) -> None:
        self.on = True

    def off_for_a_block(self) -> AbstractContextManager[None]:
        return _switched_off(self.is_on, self.switch_off, self.switch_on)


def test_a_switch_is_off_inside_the_block_and_on_again_after_it() -> None:
    switch = Switch(on=True)

    with switch.off_for_a_block():
        assert not switch.on

    assert switch.on


def test_a_switch_comes_back_on_when_the_block_fails() -> None:
    switch = Switch(on=True)

    with pytest.raises(ValueError, match="not an embedding model"), switch.off_for_a_block():
        raise ValueError("this is not an embedding model")

    assert switch.on


def test_a_switch_that_was_off_is_left_off() -> None:
    switch = Switch(on=False)

    with switch.off_for_a_block():
        assert not switch.on

    assert not switch.on


def test_the_silenced_logger_is_the_one_that_writes_the_note() -> None:
    # PyTorch's source is read, not imported: importing it takes seconds. If PyTorch
    # moves or rewords the note, this fails, and the silencing needs another look.
    torch = importlib.util.find_spec("torch")
    assert torch is not None
    assert torch.origin is not None
    (logger_name,) = local_models._NOISY_LOGGERS
    module = Path(torch.origin).parent.joinpath(*logger_name.split(".")[1:]).with_suffix(".py")

    source = module.read_text(encoding="utf-8")

    assert "logger = logging.getLogger(__name__)" in source
    assert 'logger.warning("NOTE: Redirects are currently not supported in MacOs.")' in source


def test_the_note_pytorch_logs_on_import_is_silenced() -> None:
    logger = logging.getLogger(TORCH_NOTE_LOGGER)
    try:
        quiet_model_libraries()

        assert not logger.isEnabledFor(logging.WARNING)
        # A real error from the same place would still be shown.
        assert logger.isEnabledFor(logging.ERROR)
    finally:
        logger.setLevel(logging.NOTSET)


@pytest.mark.slow
@pytest.mark.parametrize("shown_before", [True, False])
def test_progress_bars_are_left_as_they_were_found(shown_before: bool) -> None:
    from transformers.utils import logging as transformers_logging

    was_shown = transformers_logging.is_progress_bar_enabled()
    switch = (
        transformers_logging.enable_progress_bar
        if shown_before
        else transformers_logging.disable_progress_bar
    )
    try:
        switch()  # type: ignore[no-untyped-call]

        with local_models._no_progress_bars():
            assert not transformers_logging.is_progress_bar_enabled()

        assert transformers_logging.is_progress_bar_enabled() is shown_before
    finally:
        restore = (
            transformers_logging.enable_progress_bar
            if was_shown
            else transformers_logging.disable_progress_bar
        )
        restore()  # type: ignore[no-untyped-call]


LOAD_AND_EMBED = """
from handbook_rag.index.embedder import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
print(len(embedder.embed_query("How much leave do new parents get?")))
"""


@pytest.mark.slow
def test_a_model_on_disk_starts_silently_and_without_the_hub() -> None:
    from handbook_rag.index.embedder import SentenceTransformerEmbedder

    # Make sure the model is on disk. This is the one step that may download it.
    SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    # Nothing listens on port 9, so any request to the Hub would fail.
    unreachable = {**os.environ, "HF_ENDPOINT": "http://127.0.0.1:9"}
    unreachable.pop("HF_HUB_OFFLINE", None)

    run = subprocess.run(
        [sys.executable, "-c", LOAD_AND_EMBED],
        env=unreachable,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert run.stdout == "384\n"
    # No note from PyTorch, no progress bar, no notice from the Hub.
    assert run.stderr == ""
