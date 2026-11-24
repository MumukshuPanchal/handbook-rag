# Cache question pairs

Pairs of questions for choosing the similarity threshold of the semantic cache. [`pairs.jsonl`](pairs.jsonl) holds 200 of them, one on each line. Each pair puts a question of the [golden set](../golden/README.md) next to another question. In half of the pairs the other question means the same and needs exactly the golden question's answer. In the other half it only looks the same, and needs another answer or none. The task is [issue 24](https://github.com/MumukshuPanchal/handbook-rag/issues/24). What has been measured with the pairs is [further down](#what-has-been-measured-with-the-pairs), first before the cache was built and then [as it is built](#measured-again-for-the-cache-as-it-is-built).

> **Status: written and labelled by one model session, labelled again blind by a second model session, not read by a person.** Every pair, its labels and its note were written by a Claude model (`claude-opus-5-5`) in one session. A second model session labelled all 200 again without seeing those labels, and the two agree on 198 of 200 (see [the second labelling](#the-second-labelling)). No person has read the pairs. A model's check is not a person's review.

## What is in it

| Kind | What the other question needs | Pairs | Agreed |
|---|---|---|---|
| `same` | exactly the golden question's answer | 100 | 98 |
| `different` | another answer, or none, so that serving the golden question's answer would be wrong | 100 | 100 |

"Agreed" counts the pairs on which the second labelling gave the same label (`agreed: true` in the file).

The `same` pairs use 100 different golden questions, all of them questions the handbook answers (100 of the 132). The other question differs from the golden one in one of five ways, in equal parts:

| `how` | What changes | Pairs | Agreed |
|---|---|---|---|
| `reworded` | other words and another order, about the same length (median 17.5 words against 18) | 20 | 20 |
| `short` | typed as into a search box (median 7 words against 15.5) | 20 | 20 |
| `long` | a personal lead-in that does not change what is asked (median 27.5 words against 20) | 20 | 20 |
| `synonyms` | the key terms in everyday words: DMs for direct messages, exec for executive, CI minutes for compute minutes | 20 | 20 |
| `indirect` | another angle on the same fact, never a request to confirm it | 20 | 18 |

The `different` pairs, by what makes the answers differ and by how the pair was made:

| `what_differs` | What changes | `golden` | `written` | All | Agreed |
|---|---|---|---|---|---|
| `region` | another country, entity or office | 3 | 10 | 13 | 13 |
| `number` | another amount, duration, level, rank in a sequence or year | 0 | 12 | 12 | 12 |
| `entity` | another tool, team, programme, company or document | 6 | 14 | 20 | 20 |
| `negation` | the opposite: does against doesn't, downside against upside, before against after | 0 | 15 | 15 | 15 |
| `scope` | another group of people, or another case of the same rule | 0 | 15 | 15 | 15 |
| `aspect` | the same subject, another thing asked about it | 4 | 21 | 25 | 25 |
| All | | 13 | 87 | 100 | 100 |

- **131 of the 165 golden questions** are in at least one pair. The `different` pairs draw on 84 of them.
- **26 `different` pairs involve a golden question the handbook does not answer.** In 6 golden pairs one of the two questions is unanswerable. In 20 written pairs the golden question is unanswerable and the other question asks for a nearby fact that, according to the golden question's own note, the handbook does state. A cache that stored "I could not find this in the handbook." for one and served it for the other would be as wrong as one that served an answer where a decline is right.
- **22 of the 87 written questions change one word** of the golden question, 18 change two or three, and 47 change more.
- **The note of every `different` pair says which case the maker reasoned it to be.** It starts with "Differs:" when the other question has another answer (90 pairs) and with "Not answered:" when the handbook does not answer it (10 pairs), and gives the reason. The maker did not read the handbook. The second session did, and found about 12 more declines among the "Differs:" pairs (see [what the second labels rest on](#what-the-second-labels-rest-on)).

## How the pairs were made

Everything was written from the golden questions, their reference answers, their evidence quotes and, for the questions the handbook does not answer, their notes. No search, model or index was used, and nothing was looked up in the handbook.

### `same`

For each of 100 answerable golden questions, one question that a person could ask and that the reference answer answers in full. Three rules held for every pair:

- **No detail that changes the answer is added**, such as a region, a role or a year. A `long` question gets a lead-in only where the reference answer fits it: "a freelancer who signed up with a personal Gmail address" is an individual on a free email provider, the case the golden question asks about.
- **No detail the answer depends on is dropped.** A `short` question keeps the team, tool or case that the answer is about.
- **An `indirect` question asks for the same fact from another side**, never "is it X?". "How long can my laptop sit idle before it has to lock itself?" is answered by the reference answer to "How quickly should my laptop lock itself when I leave it unattended?". Two `indirect` pairs are disputed, and one of them breaks this rule (see [the two disputed pairs](#the-two-disputed-pairs)).

32 answerable golden questions have no `same` pair: those whose wording is so technical that a paraphrase risked changing what is asked (such as the rate-limiting rules of `q013` or the table order of `q093`), those the golden set's own audit flagged as answerable without the handbook or as ambiguous (`q004`, `q032`, `q042`, `q055`, `q083`), and others left out to keep the five ways at 20 each.

| Pair | Way | Golden question | Other question |
|---|---|---|---|
| `s001` | reworded | When Sales objects to a rejected downtime credit claim only to keep the customer happy, who decides whether to grant an exception? | If Sales pushes back on a rejected downtime credit claim purely to keep the customer happy, who has the final say on making an exception? |
| `s015` | short | Which label identifies issues that the Security Assurance Automation team is handling? | security assurance automation team issue label |
| `s073` | long | After I submit an emergency leave request in Workday, how quickly will the Absence Management team get back to me? | My father was taken to hospital, so I filed an emergency leave request in Workday this morning. How soon will the Absence Management team get in touch? |
| `s060` | synonyms | Which executive is the management DRI for the Board's Mergers and Acquisitions Committee? | Which exec is the management owner for the board's M&A Committee? |
| `s096` | indirect | How many weeks of parental leave do I get in the US? | What's the longest paid parental leave I can take in the US? |

### `different`

**`golden`: both questions are golden questions.** All 165 were sorted by topic, and each group was read for two questions that ask the same kind of thing about neighbouring subjects: leave and holidays by country, the German entity, the Absence Management team, Zip requests, Navan cards, the Token Management Standard, background screening, Field Marketing webcasts, the cadence of a team's meetings, and the funding of groups. 13 pairs were kept. Two kinds of pair were not used:

- Questions that share a topic but are far apart in wording. They would only add easy cases.
- Two questions the handbook does not answer, such as `u003` and `u049` on the ESPP. The assistant declines both with the same fixed sentence, so serving one reply for the other is not wrong.

**`written`: one golden question, and a question that changes one decisive detail of it.** The other question is one an employee could ask. Its answer was reasoned from the golden question's reference answer, its evidence, or its note, and the note of the pair gives that reason.

| Pair | Source | `what_differs` | Golden question | Other question | Why the answers differ |
|---|---|---|---|---|---|
| `d088` | golden | entity | How many weeks of parental leave do I get in the US? | How many weeks of paid parental leave does GitHub give its employees? (`u043`) | GitLab's 16 weeks, against another company's policy that the handbook does not hold |
| `d092` | golden | region | How many days of paid annual leave do UK team members get? | How many statutory vacation days are GitLab team members in Finland entitled to? (`u019`) | the UK's 28 days, against a country the handbook gives no figure for |
| `d080` | written | entity | How many weeks of paid parental leave does GitHub give its employees? | How many weeks of paid parental leave does GitLab give its employees? | a decline, against GitLab's 16 weeks |
| `d012` | written | negation | If a new GitLab feature doesn't use FIPS-validated crypto modules, which customers can't we offer it to? | If a new GitLab feature does use FIPS-validated crypto modules, which customers can't we offer it to? | the rule restricts only features that do not use them |
| `d019` | written | number | How many weeks of maternity leave does a GitLab France team member get when expecting twins? | … when expecting triplets? | 34 weeks is the figure for twins |
| `d053` | written | scope | Which Compass case template should a recruiter use to ask People Connect for an internal candidate's current salary details? | … for an external candidate's current salary details? | People Connect holds pay records for internal candidates only |
| `d081` | written | aspect | Is GitLab.com having an incident at the moment? (`u046`) | Where can I check whether GitLab.com is having an incident? | live data the handbook cannot hold, against where status is published, which it says |

### Changes that look decisive and are not

Some details look as if they change the answer and, by the evidence, do not. They are not in the file as `different` pairs:

- **Parental leave in another country than the US.** By `b08`'s evidence, the 16 weeks apply to any active team member, so the classic example of a false hit, US against UK parental leave, may need the same answer.
- **Another region for AWS CPPO or Google MCPO deals.** By `q066`'s evidence, the marketplaces are the distribution partners in all regions.
- **Social posts from the corporate communications team instead of a PR agency.** By `q081`'s evidence, both take the same Sprout pillar.
- **The Zoekt webserver instead of the Zoekt indexer.** By `q029`'s evidence, one document covers both.
- **More than ten years of seniority instead of more than four** for ROL in Italy. Both fall in the same band of `q063`'s evidence.

They would make useful `same` pairs that look like `different` ones, and are a reason to read the pairs a threshold gets wrong before trusting it.

## The second labelling

A second model session labelled every pair again, blind. It got the 200 pairs in a shuffled order under neutral names (`p001` to `p200`), each with only its two questions: no kind, way, note or id. It could read the golden set and the handbook snapshot, and was told not to open the maker's files. For each pair it gave a label (`same`, `different` or `unsure`), how confident it was, what the label rests on, a one-sentence reason and, where it had one, a flag. Its labels are on every line of the file as `second`, matched to the pairs through the key from its neutral names to the pair ids, and `agreed` says whether its label equals `kind`. The second session's own file of labels and that key are not in the repository. That `second` on each line is what the session wrote was checked by a script when this file was built, and cannot be checked from the repository.

| Maker's label | Second: `same` | Second: `different` | Second: `unsure` | All |
|---|---|---|---|---|
| `same` | 98 | 2 | 0 | 100 |
| `different` | 0 | 100 | 0 | 100 |
| All | 98 | 102 | 0 | 200 |

The two labellings agree on 198 of 200 pairs, and Cohen's kappa is 0.980. Both disagreements are `same` pairs that the second session called `different`. It called no `different` pair `same`, and it was unsure of none.

### The two disputed pairs

Both are `indirect`, and the second session gave both of them its lowest confidence. They stay in the file, marked `agreed: false`.

| Pair | Golden question | Other question | The maker: `same` | The second session: `different` |
|---|---|---|---|---|
| `s078` | Can I use the Context Layer's snowflake-query-guidance skill on the claude.ai website instead of the Claude Desktop app? | Do I need the Claude Desktop app to use the Context Layer's snowflake-query-guidance skill? | Both rest on one fact: the skill also works on claude.ai, and the desktop app gives the better experience. | The right reply to one opens "Yes" and to the other "No", so a stored "Yes, …" served for "Do I need the Desktop app?" reads as the wrong answer. |
| `s100` | How do I get a Navan purchasing card? | Who is eligible for a Navan purchasing card? | The reference answer to the first says which positions the cards are for, which answers the second. | The first needs the route (request through an issue, approval by Finance leadership) and the second the eligibility. A reply written for the second need not say how to request a card, so the label depends on which of the two was asked first. |

The maker accepts that `s078` breaks the rule for `indirect` questions above: "Do I need …?" is a yes-or-no question, which that rule was meant to keep out.

### Flags

The second session flagged five pairs:

| Pair | Its name | Its label | Flag |
|---|---|---|---|
| `s078` | `p079` | `different` | Both questions rest on one sentence and differ only in whether the answer is yes or no, so the label depends on whether the stored reply opens with "Yes" or "No"; a labeller judging by the fact alone would say same. |
| `s100` | `p190` | `different` | The reference answer to the first already states who is eligible, so serving it for the second would do no harm; only the reverse direction fails, and a labeller could fairly say same. |
| `d034` | `p030` | `different` | ROL is an Italian entitlement and the handbook shows no one employed through Remote in Spain, so the other question is one no employee would ask. |
| `d053` | `p093` | `different` | People Connect holds no outside candidate's salary, so the other question is one no recruiter would ask. |
| `d062` | `p031` | `different` | The reference answer to the first (`q130`) already says which view Learning and Development owns, so a stored reply to the first would also answer the second, and a hit here would do no harm. |

The flags on `d034` and `d053` question whether a person would ask the other question, not the label. The flag on `d062` is fair in one direction: a reply stored for `q130` that repeats its reference answer also answers the other question. A reply stored for the other question would not answer `q130`.

### What the second labels rest on

| `confidence` | Labels | Agreed with the maker |
|---|---|---|
| `high` | 187 | 187 |
| `medium` | 11 | 11 |
| `low` | 2 | 0 |

| `basis` | What the second session read for the label | Labels |
|---|---|---|
| `wording` | the two questions alone | 80 |
| `handbook` | the handbook snapshot | 75 |
| `golden` | the golden set | 45 |

- **The handbook check bears mostly on the `different` pairs.** 73 of the 75 labels that rest on the handbook are on `different` pairs. The `same` pairs were labelled mostly from their wording (78 of 100).
- **Its reasons correct some of the maker's notes**, which were written without the handbook. The notes are left as written, and the corrections are recorded here instead. None of them changes a label, because a decline is not the golden question's answer either.
  - `d042`: the note says the Audit Committee has a management DRI on the finance side. By the second session's reading, the board page names no management DRI for the Audit Committee, so the other question is not answered.
  - About 12 notes that start with "Differs:" are about questions that, by the second session's reading, the handbook does not answer at all: `d004`, `d010`, `d022`, `d023`, `d025`, `d027`, `d041`, `d042`, `d044`, `d054`, `d055` and `d065`. With them, about 22 of the 100 other questions in `different` pairs are declines, not 10.
  - `d016`: the note says an epic with a section:: label is not hidden by the rule in `q030`'s evidence. The second session found that such an epic still lacks the quarterly planning label, so the stored reply's reason and fix are wrong for it either way.
- **Its reasons support most of the knowledge the notes took from outside the evidence**, by its reading of the snapshot: 46 weeks for triplets and 16 weeks for one baby in France (`d019`, `d020`), up to 12 weeks of FMLA (`d090`), pro-rated leave for part-time staff in the UK (`d095`), the defence for staff who see others' pay as part of their job (`d024`), foreign requests only through a US court and a treaty (`d006`), Discover and Define as the first diamond (`d003`) and dimensions in the Common schema (`d028`). For `d018` it found Germany's maternity protection with no figure for twins, and for `d021` paternity leave of 28 days, 35 for multiple births.

## What a false hit is, and why the `different` pairs decide the threshold

A hit is a new question whose embedding is closer to a stored question than the threshold, so the stored answer is returned without a search or a request to the model. A **false hit** is a hit on a question that needs another answer: "parental leave at GitHub" served the answer stored for "parental leave in the US". The reply looks like any other answer and carries citations, which were right for the stored question, so the asker has no sign that anything went wrong. A miss costs one ordinary run of the pipeline.

Because a false hit costs far more than a miss, the threshold is chosen from the `different` pairs and then priced with the `same` pairs:

1. Use the 198 agreed pairs (98 `same`, 100 `different`), those with `agreed: true`. Report the two disputed pairs, `s078` and `s100`, apart, with their similarity, and do not count them in either rate.
2. For every pair, compute the similarity of the two questions with the embedding model the retriever uses.
3. For a range of thresholds, count the share of `different` pairs at or above it (the false-hit rate) and the share of `same` pairs at or above it (the hit rate).
4. Take the loosest threshold whose false-hit rate is acceptable, and report the hit rate there. Report both rates by `what_differs` and by `how` too, using the agreed counts in the tables above: one group, such as the 22 one-word changes, can decide the threshold on its own.

Two cautions:

- **Small counts.** If none of the 100 `different` pairs hit, the 95% interval on the false-hit rate still reaches up to 3.7%.
- **A pair is not a cache.** A real cache compares a new question with every stored one, so the chance of a false hit grows with the number of entries. These pairs measure how far apart one question and its neighbour are, not how a full cache behaves.

## What has been measured with the pairs

No cache was built when this was measured. On 24 November 2026 `handbook-rag eval-cache` measured what a cache would serve, with these pairs, and saved the run as [`evals/runs/cache/baseline`](../runs/cache/baseline). It sent no request to any model API. [The evaluation guide](../../docs/evaluation.md#a-semantic-cache-measured-before-it-is-built) gives the method, every result and what the run does not show. Every figure below is from that run.

The run follows the four steps above and goes further in three ways:

- **A second score.** Besides the similarity of the two embeddings, each pair is scored by a second model, `cross-encoder/nli-deberta-v3-small`, which reads the two questions together and gives the probability that one implies the other. It is asked in both directions, and the two are combined as the weaker direction and as the mean.
- **A simulated cache.** A pair is not a cache, as the second caution above says. So a cache that holds all 165 golden questions is simulated, the 33 that the handbook does not answer among them. Each pair's other question is asked of it, and its 3 nearest stored questions are the candidates. In the 13 pairs whose other question is a golden question itself (`source: golden`), that question's own entry is left out: serving it would be an exact match.
- **Two halves.** A threshold chosen and scored on the same pairs can flatter. Each threshold is chosen on one half of the agreed pairs, so that nothing wrong is served there, and the result is reported on the other half: 99 pairs on each side, 49 `same` and 50 `different`. Pairs that name the same golden question stay on one side, because the pairs are not independent (see [Known limits](#known-limits)). The split balances `same` against `different`, and nothing else.

What it found, in short:

- **The similarity of the embeddings alone does not tell the two kinds apart.** Its area under the curve is 0.546, where 0.5 is a coin, and the highest similarity of all 198 agreed pairs belongs to a `different` pair. No threshold was safe on the choosing half, so on the reporting half it served nothing.
- **The second model separates them far better**: 0.865 by its weaker direction and 0.915 by the mean. At the threshold the choosing half gave, it served 7 of the 49 `same` questions of the reporting half right, and none of the 50 `different` ones.
- **None of 50 was no test of that threshold.** The highest score at which a `different` pair of the reporting half would be served is 0.143 by the weaker direction, so the same result would have come out at almost any threshold. The 3 `different` pairs that the second model scores above 0.8 are all on the choosing half. The interval on none of 50 reaches 0.071, and it is too narrow: the 50 come from 31 groups of pairs that name the same golden question. On 200 other splits of the same pairs, the same rules served something wrong on the reporting half in 97, and those are exactly the splits that put `d067` there.
- **The two disputed pairs are in no figure.** `s078` has a similarity of 0.963, and an entailment of 0.975 in one direction and 0.035 in the other. `s100` has a similarity of 0.893, and 0.015 and 0.065. Neither is served at any of the chosen thresholds.

By `how`, as step 4 asks. The medians are over all agreed pairs. The last four columns are the two halves, decided by the weaker direction of the second model:

| `how` | Agreed pairs | Median similarity | Median entailment, weaker direction | Median entailment, mean | On the choosing half | Served right there | On the reporting half | Served right there |
|---|---|---|---|---|---|---|---|---|
| `reworded` | 20 | 0.964 | 0.989 | 0.992 | 15 | 8 | 5 | 2 |
| `short` | 20 | 0.893 | 0.007 | 0.490 | 11 | 0 | 9 | 0 |
| `long` | 20 | 0.906 | 0.039 | 0.516 | 10 | 0 | 10 | 0 |
| `synonyms` | 20 | 0.907 | 0.912 | 0.949 | 7 | 1 | 13 | 5 |
| `indirect` | 18 | 0.889 | 0.001 | 0.097 | 6 | 1 | 12 | 0 |

By `what_differs`, over all agreed pairs, with the reporting half in the last two columns:

| `what_differs` | Agreed pairs | Median similarity | Highest similarity | Median entailment, weaker direction | Highest entailment, weaker direction | On the reporting half | Served wrongly there |
|---|---|---|---|---|---|---|---|
| `region` | 13 | 0.912 | 0.980 | 0.000 | 0.821 | 10 | 0 |
| `number` | 12 | 0.938 | 0.994 | 0.000 | 0.143 | 7 | 0 |
| `entity` | 20 | 0.849 | 0.918 | 0.000 | 0.495 | 9 | 0 |
| `negation` | 15 | 0.934 | 0.992 | 0.000 | 0.005 | 7 | 0 |
| `scope` | 15 | 0.969 | 0.991 | 0.001 | 0.134 | 8 | 0 |
| `aspect` | 25 | 0.840 | 0.962 | 0.000 | 0.989 | 9 | 0 |

By `source`, over all agreed `different` pairs, with the number on each half:

| `source` | Agreed pairs | Lowest similarity | Median | Highest | Highest entailment, weaker direction | On the choosing half | On the reporting half |
|---|---|---|---|---|---|---|---|
| `golden` | 13 | 0.497 | 0.685 | 0.790 | 0.495 | 4 | 9 |
| `written` | 87 | 0.673 | 0.918 | 0.994 | 0.989 | 46 | 41 |

- **A single pair decides each threshold.** Step 4 warns that one group can. For the similarity it is `d012`, which changes one word, "doesn't" to "does", and has the highest similarity on the choosing half. For the second model it is `d067`, an `aspect` pair that asks for which quarters a budget was allocated where its golden question asks how much.
- **The two `different` pairs the second model scores highest are stored declines.** `d067` and `d081` are the only ones it scores above 0.9. In both, the golden question is one the handbook does not answer (`u004` and `u046`), so what a cache holds for it is a decline, and by the pair's note the other question has an answer. A cache that stores no declines would have met neither of them. That was not measured. It was measured afterwards, for [the cache as it is built](#measured-again-for-the-cache-as-it-is-built).
- **The second model passes plain rewordings, and few others.** On the reporting half it served 2 of the 5 `reworded` and 5 of the 13 `synonyms` questions, and none of the 31 `short`, `long` and `indirect` ones. On the choosing half it served one of the 27 of those three kinds, `s062`. Most of them have at least one direction of entailment that is low, as the medians show.
- **The 13 `golden` pairs are not close.** Two golden questions on neighbouring subjects are far apart for a near-miss, as the table by `source` shows, and 9 of them are among the 50 `different` pairs of the reporting half. The hard cases are among the 87 `written` pairs.
- **The set is hard on purpose.** Each rate is per `different` question asked or per `same` question asked, not per question a real user asks. Whether a threshold chosen here is stricter than real use needs is not measured.

## Measured again for the cache as it is built

The cache was built after the measurement above ([ADR 0020](../../docs/adr/0020-an-answer-cache-that-matches-by-meaning.md)). It keeps a decline for the very question that was declined and never offers it to another question. So `handbook-rag eval-cache --no-declines` measured the pairs once more with that one change, on 24 November 2026, and saved the run as [`evals/runs/cache/no-decline-candidates`](../runs/cache/no-decline-candidates). [The evaluation guide](../../docs/evaluation.md#the-answer-cache-measured-as-it-is-built) gives every result. Every figure below is from that run.

- **What changed.** The 33 stored questions that the handbook does not answer are left out of every question's candidates, and each question meets the 3 nearest of the other 132. That such a question holds a decline is the golden set's label, and no model was asked. Pairs, split, rule and models are those of the first run.
- **The threshold is lower: 0.8669 by the weaker direction.** It is set by `d004`, a `region` pair, at 0.8206. The cache's default threshold is this number.
- **On the reporting half it served 16 of the 49 `same` questions right, none of the 50 `different` ones, and no `same` question another stored question's answer.**
- **That half held nothing hard near the threshold.** The highest score at which a `different` pair of it would be served is 0.143. A `same` pair, `s073`, would be served another stored question's answer up to 0.854.
- **On 200 other splits of the same pairs the rule served something wrong on the reporting half in 91.**

## Format

Each line is one JSON object with these fields:

| Field | Value |
|---|---|
| `pair` | `s001` to `s100` for `same`, `d001` to `d100` for `different` |
| `kind` | `same` or `different`: the maker's label |
| `question_id` | the golden question's id |
| `question` | its text, word for word |
| `other` | the other question |
| `other_id` | for a `golden` pair, the other golden question's id; otherwise `null` |
| `how` | for `same`: `reworded`, `short`, `long`, `synonyms` or `indirect`; otherwise `null` |
| `what_differs` | for `different`: `region`, `number`, `entity`, `negation`, `scope` or `aspect`; otherwise `null` |
| `source` | for `different`: `golden` or `written`; otherwise `null` |
| `note` | one sentence: for `same`, why the same answer is right; for `different`, what the other question's answer is or why the handbook does not answer it |
| `second` | the second labelling: `label` (`same`, `different` or `unsure`), `confidence` (`high`, `medium` or `low`), `basis` (`wording`, `handbook` or `golden`), `reason`, and `flag` only when it raised one |
| `agreed` | `true` when the second labelling's `label` equals `kind` |

```json
{"pair": "d012", "kind": "different", "question_id": "q021",
 "question": "If a new GitLab feature doesn't use FIPS-validated crypto modules, which customers can't we offer it to?",
 "other": "If a new GitLab feature does use FIPS-validated crypto modules, which customers can't we offer it to?",
 "other_id": null, "how": null, "what_differs": "negation", "source": "written",
 "note": "Differs: q021's evidence restricts only features that do not use FIPS-validated modules, so a feature that does is not kept from any customer by this rule.",
 "second": {"label": "different", "confidence": "high", "basis": "golden",
            "reason": "The negation is flipped: the handbook bars only features that do not use FIPS-validated modules from FIPS-release and Dedicated for Government customers, so the second has no restriction to report."},
 "agreed": true}
```

## Known limits

- **Labelled by two model sessions and read by no person.** A model's check is not a person's review. The second session labelled blind, but both are model sessions and can share blind spots.
- **Built on questions that were mostly written by a model that could see the passage.** 122 of the 132 answerable golden questions were. The other questions were written by a model too. Real users' wording, with typos, vaguer questions and several questions at once, is not represented, and model-written paraphrases may sit closer together, or further apart, than real ones.
- **The maker reasoned the answers of the `written` questions and did not look them up.** They come from the reference answer, the evidence quote or the golden question's note. The second session read the handbook snapshot for 73 of the `different` pairs and agreed with every label, but its reading corrects several notes (see [what the second labels rest on](#what-the-second-labels-rest-on)).
- **The pairs are not independent.** The 200 pairs use 131 golden questions, 59 of them more than once, and four questions (`q034`, `q063`, `b13` and `b08`) are in 19 pairs. A fault in one golden question, or a quirk of its wording, moves several pairs at once, and an interval computed as if the 200 pairs were independent is too narrow.
- **The set is hard by construction.** By plain word overlap (the share of distinct words two questions have in common), the `different` pairs are closer to their golden question than the `same` pairs: medians of 0.63 against 0.45. The second session, counting its own way, found 0.61 against 0.43. Real traffic is not half near-misses, so a rate measured here is per near-miss asked, not per question asked. Whether a threshold chosen here is stricter than real use needs is not measured.
- **Style predicts the label.** All 20 questions typed as into a search box are `same`, and no `different` question is a fragment. Of the 22 other questions that open with a personal lead-in, 20 are `same`; the 2 `different` ones are `d045` and `d056`, and in `d056` the golden question has the same lead-in. The set cannot show whether a short fragment is matched to a neighbouring question it should not be.
- **Declines are tested one way only: against an answer.** In 30 `different` pairs one question should be declined and the other answered: 26 in which a golden question is unanswerable and 4 in which the written question is the one to decline (by the second session's reading, about 12 more). No pair has two questions that should both be declined, and no `same` pair rewords an unanswerable question, so whether the cache may reuse a stored decline for a reworded unanswerable question is not tested.
- **Some labels depend on direction, or on how the stored reply happens to be worded.** The second session named about seven: `s011`, `s064`, `s066`, `s083`, `s078`, `s100` and `d062`. In each, a reply to one question would serve the other in one direction and fall short in the other, or a reply would open with a bare "Yes" or "No" that reads wrongly against the other question. The cache serves in one direction, the reply stored for the earlier question given for the later one, and a single label per pair cannot hold that.
- **English only**, and single questions without a conversation around them.
- **200 pairs.** With 100 of each kind, a rate is known to within several points at best (see the cautions above).
- **Uneven coverage.** Some golden questions are in several pairs (`q034` in six, `q063` and `b13` in five each), so their topics weigh more. The `what_differs` groups are uneven too, from 12 `number` pairs to 25 `aspect` pairs, and where two labels fit, one was chosen: `d093` differs both in region and in what is counted, and is labelled `region`.
- **A snapshot in time.** The pairs rest on the golden set, which is tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`. A `same` pair stays the same when the handbook changes, but a question the handbook does not answer today may be answered tomorrow, which would turn a "Not answered:" note into "Differs:".

## Licence and attribution

The golden questions in the pairs (`question`, and `other` in the `golden` pairs) are those of the [golden set](../golden/README.md), some of which repeat a few words of the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The file holds no passage of the handbook. The other questions and the notes were written for this project. The second labelling's reasons name handbook pages and give short facts from them; in 11 of them a run of six or more words is the same as in an evidence quote of the golden set, and none holds a whole passage.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
