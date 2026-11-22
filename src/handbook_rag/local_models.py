"""Load the open-weight models that run on this machine.

The embedding model and the reranker are downloaded from the Hugging Face Hub the first
time they are used, and kept in a cache on disk. Left to themselves, the libraries still
ask the Hub on every start whether the files have changed. Every command then waited for
a network round trip, printed the Hub's notice about anonymous requests, and needed the
Hub to be reachable, all for files that were already on disk.

So a model is read from disk first, and the Hub is asked only when it is not there. One
consequence: a model does not update itself. To fetch a newer revision, delete the
model's folder from the cache.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

# PyTorch logs a note from this module each time it is imported on macOS. The note is
# about a feature this program does not use.
_NOISY_LOGGERS = ("torch.distributed.elastic.multiprocessing.redirects",)


def quiet_model_libraries() -> None:
    """Silence start-up messages that say nothing about this program.

    Call it before PyTorch is first imported: the note is logged during the import.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def load_local_first[T](load: Callable[..., T], model_name: str, **options: object) -> T:
    """Load a model from the download cache, and from the Hub only when it is not there.

    ``load`` is a model class from sentence-transformers. It is called with the model's
    name and ``options``.
    """
    try:
        with _no_progress_bars():
            return load(model_name, local_files_only=True, **options)
    except OSError:
        # Not downloaded yet, or only in part. Asking the Hub fetches what is missing,
        # and this time the progress of the download is shown.
        return load(model_name, **options)


def _no_progress_bars() -> AbstractContextManager[None]:
    """Hide the bar that counts a model's weights as they are read from disk.

    For the embedding model the bar is over in under a tenth of a second, so it only
    adds a line to every command.
    """
    from transformers.utils import logging as transformers_logging

    return _switched_off(
        transformers_logging.is_progress_bar_enabled,
        transformers_logging.disable_progress_bar,
        transformers_logging.enable_progress_bar,
    )


@contextmanager
def _switched_off(
    is_on: Callable[[], bool], switch_off: Callable[[], object], switch_on: Callable[[], object]
) -> Iterator[None]:
    """Switch something off for a block, and leave it afterwards as it was found.

    It comes back on even when the block fails, and stays off if it was off before.
    """
    was_on = is_on()
    switch_off()
    try:
        yield
    finally:
        if was_on:
            switch_on()
