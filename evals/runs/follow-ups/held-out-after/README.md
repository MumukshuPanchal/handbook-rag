# held-out-after: the held-out replies, read again after the rule was changed

This is not a run of its own. It holds the replies of [`held-out`](../held-out), copied, read again with no request under the rule for a rewrite without a question mark as it stands, after the requirement for an opening was taken out on 25 November 2026, once the held-out replies had been seen ([ADR 0024](../../../../docs/adr/0024-follow-up-questions-in-the-service.md)).

- `replies.jsonl` and `run.json` are copies of the held-out run's, byte for byte. So `run.json` and `summary.json` count the 20 requests that run made: none was made for this folder.
- `summary.json` and `results.jsonl` were written by `handbook-rag eval-follow-ups --cases evals/follow-ups/held-out/cases.jsonl --variant held-out-after`, from those replies, with the code of today. The description in `summary.json` is the command's own, the same as the held-out run's.
- `readings.jsonl` holds the held-out run's readings, and 2 more, of the 2 replies that the rule as it stood before refused, read as if the step had used them.

What it gives is not held-out evidence: the rule it is read under was changed after these replies were seen.
