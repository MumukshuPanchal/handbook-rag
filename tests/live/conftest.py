"""What every live test shares: it measures the real model, so nothing stored may answer."""

from collections.abc import Iterator

import pytest

from handbook_rag.measuring import measuring_the_model


@pytest.fixture(autouse=True)
def nothing_stored_stands_in() -> Iterator[None]:
    """Run each live test with the answer cache switched off for the whole process.

    A live test spends a real request to see what the model does today. An answer from
    the cache would pass it without a request. While the switch is on, a cache cannot be
    built and cannot be handed to the entry point of an answer.
    """
    with measuring_the_model():
        yield
