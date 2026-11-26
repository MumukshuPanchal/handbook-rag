# What to decide for each reply

Each line of `items.jsonl` is one reply of an assistant that answers questions from a company
handbook and cites its sources. For each you get:

- `question`: what the person asked;
- `passage_title` and `passage_text`: the one passage the assistant was given. The title is
  the page's title with the headings above the passage, and it is part of what the passage
  says;
- `limit`: a limit that this passage puts on its rule, in the passage's own words (who the
  rule is for, where or when it holds, how much, on what condition, with what exception),
  and `limit_where`: whether the passage states it in its `text` or in its `title`;
- `reply`: the reply as the person saw it. `[1]` is a citation of the passage.

Every question comes twice in the file, each time with another reply. Decide each reply on
its own; do not look for the other one.

Label each reply with one of:

- `kept`: the reply gives the limit, in any wording, anywhere in it; or the question itself
  asks about that very group, place, time, amount or case, so that the reply need not
  repeat it.
- `left out`: the reply gives the rule and not the limit, so that a reader would take the
  rule to hold more widely than the passage says.
- `no answer`: the reply declines, or does not answer the question.

A weaker wording of the limit (say "new team members" where the passage says "in their
first year of employment") is labelled by whether a reader would take the rule to cover a
case the passage excludes; mark such a label `"unsure": true`. Mark any other label you
would not defend against a careful second reader as unsure too.

Only the named limit counts. Do not judge whether the rest of the reply is right, complete
or well written.

Write one line for each item, in the order of the file:
`{"item": "a01", "label": "kept" | "left out" | "no answer", "unsure": true | false, "note": "one sentence that quotes the words of the reply your label turns on, or says what is missing"}`.
