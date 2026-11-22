"""Keep one stage of an evaluation run on disk, so that it can be continued.

Asking the questions is one stage of a run, and each judged criterion is another. Every
stage sends requests that cost a share of a daily allowance. So each one saves a row as
soon as it has it, records the settings that made its rows, and counts its requests
before it sends them. This module holds what the stages share.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SavedRunError(ValueError):
    """A saved run cannot be continued: other settings made it, or its file cannot be read."""


@dataclass(frozen=True, slots=True)
class Stage:
    """The files of one stage in a run directory."""

    run_dir: Path
    rows_file: str
    """One saved row on each line, as JSON."""
    state_file: str
    """The settings that made the rows, and the count of attempts."""
    holds: str = "replies"
    """What a row is, for messages."""
    advice: str = "Give this run another name, or discard its saved replies with --restart."
    """What to do when the rows were made with other settings."""
    outputs: tuple[str, ...] = ()
    """Other files the stage writes, removed with its rows when it is restarted."""
    traces: bool = False
    """Whether the stage writes one file per question into ``traces/``."""

    @property
    def rows_path(self) -> Path:
        return self.run_dir / self.rows_file

    @property
    def state_path(self) -> Path:
        return self.run_dir / self.state_file

    def open(
        self, config: Mapping[str, object], *, restart: bool = False
    ) -> list[tuple[int, dict[str, Any]]]:
        """Start the stage, or pick it up where it stopped. Returns the rows already saved.

        Rows made with other settings are never mixed in. The caller chooses: keep them
        and run under another name, or ``restart`` to discard what the stage holds. A
        stage that holds no row yet simply takes the new settings, since there is nothing
        to mix.

        A line cut short at the end of the saved rows, left by a run that was killed
        while it wrote, is removed here. Left in place, the next row would be written
        onto it.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        wanted = _as_saved(config)
        if restart:
            self.discard()
        _remove_cut_line(self.rows_path)
        saved = self._rows_to_continue(wanted)
        if not self.state_path.is_file():
            write_json(self.state_path, {"config": wanted, "attempts": 0})
        else:
            state = self._state()
            if state["config"] != wanted:
                write_json(self.state_path, {**state, "config": wanted})
        return saved

    def check(self, config: Mapping[str, object]) -> None:
        """Refuse a stage that these settings cannot continue. Reads only, writes nothing.

        It is for refusing early, before models are loaded, which takes a while. A stage
        that passes is opened later with :meth:`open`.
        """
        self._rows_to_continue(_as_saved(config))

    def append(self, row: Mapping[str, Any]) -> None:
        """Save one row at once, so that a run that stops loses nothing it paid for."""
        with self.rows_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def add_attempts(self, attempts: int = 1) -> int:
        """Add to the stage's count of attempts, and return the new total.

        The count is written before each request is sent, not after the run: a run that
        is interrupted has then still counted every request it sent.
        """
        state = self._state()
        state["attempts"] += attempts
        write_json(self.state_path, state)
        return int(state["attempts"])

    def attempts(self) -> int:
        """How many requests the stage may have sent since it was started or restarted."""
        return int(self._state()["attempts"])

    def config(self) -> dict[str, Any]:
        """The settings the stage's rows were made with."""
        return dict(self._state()["config"])

    def discard(self) -> None:
        """Remove everything the stage has written, so that nothing outlives a restart."""
        for name in (self.rows_file, self.state_file, *self.outputs):
            (self.run_dir / name).unlink(missing_ok=True)
        if self.traces:
            clear_traces(self.run_dir)

    def _rows_to_continue(self, wanted: Mapping[str, object]) -> list[tuple[int, dict[str, Any]]]:
        """The rows the stage holds, once it is clear that these settings may continue it."""
        saved = load_rows(self.rows_path)
        if not self.state_path.is_file():
            if saved:
                raise SavedRunError(
                    f"{self.rows_path} holds {self.holds}, but nothing records the settings "
                    "that made them. Give this run another name, or discard them with "
                    "--restart."
                )
            return saved
        stored = self._state()["config"]
        if saved and stored != wanted:
            changed = sorted(
                key for key in {*stored, *wanted} if stored.get(key) != wanted.get(key)
            )
            raise SavedRunError(
                f"the run in {self.run_dir} was made with other settings "
                f"({', '.join(changed)}). {self.advice}"
            )
        return saved

    def _state(self) -> dict[str, Any]:
        """What the stage recorded about itself: its settings and its count of attempts."""
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
                raise TypeError("it does not hold a run's settings")
            attempts = state.get("attempts", 0)
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
                raise TypeError("its count of attempts is not a whole number")
        except (ValueError, TypeError) as error:
            raise SavedRunError(
                f"{self.state_path} cannot be read ({error}). Discard the run with --restart."
            ) from error
        return {**state, "attempts": attempts}


def load_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read saved rows, each with the number of its line.

    Every row is one line that ends in a line break. Text after the last line break is a
    line that a killed run did not finish, and is left out: that row is made again. A
    finished line that cannot be read means the file was edited or damaged, which is
    reported.

    Lines are split at line breaks only. A row may hold other characters that some tools
    treat as the end of a line, and splitting there would cut it in two.
    """
    if not path.is_file():
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    finished = path.read_text(encoding="utf-8").split("\n")[:-1]
    for number, line in enumerate(finished, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SavedRunError(f"{path}:{number} cannot be read: {error}") from error
        if not isinstance(row, dict):
            raise SavedRunError(f"{path}:{number} is not a saved row.")
        rows.append((number, row))
    return rows


def clear_traces(run_dir: Path) -> None:
    for trace in (run_dir / "traces").glob("*.json"):
        trace.unlink()


def write_json(path: Path, value: object) -> None:
    _replace(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _as_saved(config: Mapping[str, object]) -> dict[str, Any]:
    """Settings in the shape they have after being saved and read back."""
    saved: dict[str, Any] = json.loads(json.dumps(dict(config)))
    return saved


def _remove_cut_line(path: Path) -> None:
    """Cut the file back to its last finished line."""
    if not path.is_file():
        return
    saved = path.read_bytes()
    if saved and not saved.endswith(b"\n"):
        _replace(path, saved[: saved.rfind(b"\n") + 1])


def _replace(path: Path, content: bytes) -> None:
    """Write a file in one step, so that a kill never leaves half of it."""
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(content)
    os.replace(partial, path)
