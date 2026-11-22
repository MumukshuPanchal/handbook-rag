"""Real requests to the Gemini API.

Every other test uses a stand-in for the API, so none of them can notice that the API or
its SDK has changed. These can. They need a key and use up requests, so they do not run
by default and never run in CI: use ``make test-live``.
"""

from pathlib import Path

import pytest

from handbook_rag.config import Settings
from handbook_rag.wiring import make_generator
from tests.support import make_chunk

pytestmark = pytest.mark.live

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
EXPENSES = make_chunk(
    "A laptop can be expensed up to one thousand dollars.",
    doc_id="finance/expenses.md",
    title="Expenses",
)


@pytest.fixture
def settings() -> Settings:
    # Tests run in an empty directory, so the project's own .env file is named here. The
    # model is the default one, whatever that file selects for everyday use.
    settings = Settings(
        _env_file=PROJECT_ROOT / ".env",
        answer_model=Settings.model_fields["answer_model"].default,
    )
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        pytest.skip("no GEMINI_API_KEY in the project's .env file")
    return settings


def test_gemini_answers_from_the_passages_and_cites_the_right_one(settings: Settings) -> None:
    received: list[str] = []

    answer = make_generator(settings).generate(
        "How long is parental leave?", [EXPENSES, LEAVE], on_text=received.append
    )

    assert answer.answered
    assert "sixteen" in answer.text.lower() or "16" in answer.text
    assert answer.sources == (LEAVE,)
    assert "".join(received) == answer.text
    assert answer.stop_reason == "end_turn"
    assert answer.model.startswith("gemini-")
    assert answer.usage.input_tokens > 0
    assert answer.usage.output_tokens > 0


def test_gemini_declines_when_the_passages_do_not_hold_the_answer(settings: Settings) -> None:
    answer = make_generator(settings).generate(
        "What is the wifi password in the Berlin office?", [EXPENSES, LEAVE]
    )

    assert not answer.answered
    assert answer.sources == ()
