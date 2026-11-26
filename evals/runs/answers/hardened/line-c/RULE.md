# What to decide for each reply

Each line of `items.jsonl` is one reply of an assistant that answers questions from a company
handbook and cites its sources. The product changed the reply before showing it: a check
declined it, or it was read as a form that says the passages do not answer the question and
was shown as the sentence "I could not find this in the handbook." For each line you get
the `question`, the `reference_answer` (the right answer, written beforehand), the
`model_words` (what the model itself wrote), the `reply_as_shown` (what the person saw),
and `done` (what the product did).

Decide one thing: **did the person lose the answer?** That is so when the reply as shown
does not give the fact that answers the question, and the model's own words did.

- The model's words give the fact if they state what the question asks for as the reference
  answer states it. Other wording is fine.
- Where a reference answer holds several facts, the one that answers the question's main
  clause is the one that counts.
- If the model's words say that the passages do not say, or do not answer, they do not give
  the fact.
- In doubt, the reply counts as lost.

Write one line for each item: `{"question_id": "...", "lost": true | false, "unsure": true |
false, "note": "one sentence that quotes the words your decision turns on"}`.
