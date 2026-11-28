# Changelog

What each version of this project contains. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each item names the pull request or decision record that holds it.

## [Unreleased]

### Fixed

- The README, the roadmap, the evaluation guide and the results of the search variants sent the step that splits a question asking for two things to [#79], which closed with the rewriting of follow-up questions and without it. They now send it to [#22], query rewriting before retrieval, and the decision record has a dated note that says so ([#92], [ADR 0021]).

## [1.1.0] - 2026-11-27

The second version: the assistant takes follow-up questions, serves an answer it wrote before to the same question or to one that a second model judges to mean the same, and answers under version 2 of its instructions, with checks of each reply in code, against instructions planted in a page. The API serves a demo page that holds a conversation, and an MCP server offers the search to AI assistants. Version 2 was measured against version 1 on the same questions and on cases kept from the work on it, and the [README](README.md#results) reports what got better and what got worse, such as fewer declines of the questions the handbook does not answer.

### Added

- `index --reuse-from <older commit>`: index a newer commit of the handbook into a collection of its own, taking the vectors of every chunk whose embedded text is unchanged from the older commit's collection, which is only read. It refuses vectors made with another embedding model, keyword model or chunking, and reports the pages added, changed, removed and unchanged, the chunks reused and embedded, and how long each part took ([#21], [ADR 0019]).
- `check-reuse`, which indexes the retrieval gate's slice of a newer commit twice in memory, in full and by reuse, compares the two, and fails when they differ or when nothing was reused, and `compare-collections`, which compares two collections in the store: ids, payloads, and vectors within a tolerance ([#21], [ADR 0019]).
- `handbook-rag mcp`: handbook search for AI assistants and coding tools, as one read-only tool over the Model Context Protocol, on standard input and output. It runs the same retriever as `POST /search`, checks its arguments with the same limits and returns the same object, and the passages as text too. No tool answers, so a call spends no request of the answer model ([#20], [ADR 0017]).
- A demo page at `/`, served by the API: a question box that shows the answer as the model writes it, then the answer with its sources, the model, the tokens, the timings and the trace id, and beside it what the search found. Three hand-written files with no build step. Everything from the API is shown as text, an address in an answer is never a link, and a strict content security policy is sent with each file. `HANDBOOK_RAG_API_DEMO=false` switches it off ([#19], [ADR 0016]).
- Time limits for an answer asked for on the command line or over HTTP: one request to the model may be silent for 20 seconds, and the model may be tried again for 30 (`HANDBOOK_RAG_ANSWER_TIMEOUT_SECONDS`, `HANDBOOK_RAG_ANSWER_DEADLINE_SECONDS`). The repeats are made by our own code instead of the SDKs, three attempts at most: each one is counted in the answer's new `requests` and has a span of its own, and no new attempt is decided on after the deadline: a pause is cut short to end at it, and the attempt after that pause starts at the deadline. What is tried again is decided by the kind of failure, and nothing is tried again once text has been sent. On an overloaded evening a silent model was given up after two requests and 42 seconds ([#42], [ADR 0018]).
- An optional fallback model, `HANDBOOK_RAG_ANSWER_FALLBACK_MODEL`, asked once when the answer model gives up for any reason but a wrong setting. The answer's new `fallback` says when it wrote the reply. A missing key for its provider is found when the service starts. A fallback that succeeds has not been seen for real yet ([#42], [ADR 0018]).
- `POST /ask` has a 503 of its own for a model that did not answer in time, and its JSON object, also in the `answer` event of a stream, has `requests` and `fallback`. The demo page shows both when an answer took more than one request ([#42], [ADR 0018]).
- `handbook-rag eval-cache` and `make eval-cache`: a measurement of what a semantic cache would serve, made before the cache was built. It uses 200 labelled pairs of a golden question and a rewording or a near-miss (`evals/cache/`), simulates a cache that holds the golden questions, chooses each threshold on one half of the pairs and reports on the other. It sends no request to a model API. On the first run the similarity of two embeddings alone could serve nothing safely, and a second check served 7 of 49 rewordings and none of 50 near-misses on the half it had not seen, a half that held no near-miss hard enough to test its threshold. No person has read the pairs ([#24], [the guide](docs/evaluation.md#a-semantic-cache-measured-before-it-is-built)).
- `handbook_rag.cache.verify`: the second check on its own, which the answer cache uses. A cross-encoder trained for natural language inference reads a stored question and a new one in both directions. Which of its outputs means entailment is read from the model's configuration. `HANDBOOK_RAG_CACHE_VERIFIER_MODEL` names the model ([#24]).
- An answer cache, on by default in the mode `semantic` by the decision of the project's owner (`HANDBOOK_RAG_CACHE_MODE`: `semantic`, `exact` or `off`). An answer the answer model wrote in full is stored in a Qdrant collection of its own and served again without a request: to the same question, where case, spacing and the marks at its end do not count, and to a question that the second check judges to mean the same. A stored decline, an answer in which one of two guards finds that part of the question was declined, and an answer that cites nothing are served to the same question only. The second guard is a list of phrases, fitted to four replies of the saved answer run: it errs towards keeping an answer out, and a reply that says it in other words gets past it. Of 5 replies it was not written from, it finds 1. The collection's name changes with every setting that decides an answer, and a version number that a person raises stands for the code beside them. When `index` has changed the index and finished, it empties every cache of that handbook commit. A lookup waits 2 seconds at most, and then the model is asked. What a process does only once is not counted, and a lookup that was given up does not take a turn at the second check. A question that only looks like a stored one will sometimes be served its answer, so a cached answer says that it is one and names the question it was written for: in `ask`, in the new `cached` field of `POST /ask`, also in a stream, and on the demo page. `ask --no-cache` and `"cache": false` get a fresh answer. An evaluation never gets a cache: while a command that measures runs, none can be built ([#24], [ADR 0020]).
- `handbook-rag eval-cache --no-declines` runs the measurement with stored declines left out of the candidates, by the golden set's label (the saved run `evals/runs/cache/no-decline-candidates`). `eval-cache --replies` measures the cache as it is built: the simulated cache is filled with the replies of a saved answer run, and the cache's own code decides which of them may be served by meaning (the saved run `evals/runs/cache/as-built`). On the half of the pairs that chose nothing that run served 16 of 49 rewordings and nothing wrong, on a half that again held nothing hard near the threshold. Its rule gives 0.8669. The cache's default threshold is 0.91 (`HANDBOOK_RAG_CACHE_THRESHOLD`), the top of the range that serves the run the same, chosen with every pair in view. The default floor is 0.7 (`HANDBOOK_RAG_CACHE_FLOOR`), which saves the second check work and does nothing for correctness, and the cache checks 3 candidates (`HANDBOOK_RAG_CACHE_CANDIDATES`). The defaults are fitted on pairs no person has read, and nothing is measured on real questions ([#24], [the guide](docs/evaluation.md#the-answer-cache-measured-as-it-is-built)).
- `handbook-rag cache-status` says how many answers the cache holds, and `handbook-rag cache-clear` empties it. With `--all` it also removes the caches left behind under other settings, and only collections named as this project names them. The lookup is a span of its own, `cache.lookup`, with the tier, the candidates and their scores, and no question unless content is on ([#24], [ADR 0020]).
- When no answer can be written, `ask` says what failed and lists the passages the search found. When an answer took more than one request, its last line says how many, and whether the fallback model wrote it ([#42]).
- The two-passage questions, `evals/two-passage/`: 34 questions whose answer needs two passages on two different pages, each with a reason for either passage, and the lines of a second reading by three sessions of the model that wrote them, which confirm 33. `golden-check --reasons` holds the file to two passages on two pages for each question, a reason for each passage, and a review that fits what the second reading found. A model wrote the questions with both passages in front of it, no person has read them, and no answer was written for them ([#78], [ADR 0021]).
- `eval-retrieval` scores All@3, All@5 and All@10 beside Hit@k: whether the first results hold every passage a question needs, where a hit means that one of them was found. A row also records the rank at which each passage came and the rank of the first chunk of its page, and the summary counts the questions with all, some and none of their passages found. `--reviewed-only` leaves out a question whose review is pending. For a question with one passage All@k is Hit@k, and the saved runs of the golden set are as they were. The default search has both passages in its first five results for 12 of the 33 confirmed questions, and at least one for 26 ([#78], [ADR 0021]).
- Variants of hybrid search for one run of `eval-retrieval`, each off unless it is given: `--spread-per-page` with `--spread-within` for at most so many chunks of one page among the first results (`handbook_rag.retrieve.spread`), which no setting and no other command has, and `--fusion-candidates` and `--rrf-k`, which change for one run what `HANDBOOK_RAG_FUSION_CANDIDATES` and `HANDBOOK_RAG_RRF_K` set for every command. The command refuses a variant it could not measure. Fourteen variants were run on the 33 questions and on the golden set, and none is adopted: none finds both passages clearly more often, and the limit for each page takes Hit@5 on the golden set from 0.962 to 0.803 with one chunk allowed and to 0.894 with two. The default search, the settings and the retrieval gate are as they were ([#78], [ADR 0021]).
- The rewriting step, `handbook_rag.rewrite`: a follow-up question is rewritten into one that stands alone, with one request to the answer model, from the questions of the earlier turns and never their answers. When the reply does not pass the step's checks of one question, or the request fails, the question is searched for as it was asked, and the step says so. It runs in front of an answer since the item below ([#79], [ADR 0022]).
- `handbook-rag eval-follow-ups` and `make eval-follow-ups`: 82 labelled conversations (`evals/follow-ups/`), read again by two more model sessions, put to the step with the real model, one request each. Code decides what it can, the answer cache's verifier takes a rewrite that reaches 0.99 in both directions, and a reader decides the rest. On the run, 35 of 43 follow-ups got a question that means an acceptable form, one of them a close call, 16 of 17 questions that stand alone were left alone, and none of the 22 planted instructions was followed. No answer was written for the rewritten questions, and no person has read the cases ([#79], [ADR 0022]).
- Labelled answers for one fault, `evals/judge/limits/`: 120 answers on 40 passages that keep the limit their passage states, leave it out, or leave out something harmless. One model session wrote and labelled them, and three reader sessions labelled them again without seeing those labels, by the account of how they were briefed: the two labellings agree on 196 of 197 statements. No person has read them. `make eval-judge-limits` checks a judge on them ([#59], [ADR 0023]).
- Version 2 of the support judge, `--judge-version 2` on `eval-judge` and `eval-support`. It is told the whole rubric and how to look for a limit that an answer leaves out, and it names the limits it found before its verdict. On the 120 limit answers it caught 33 of the 39 limits left out, where version 1 caught 13, with one false alarm in 157 statements. On the first labelled answers it caught 40 of 41 faults with 2 false alarms, where version 1 caught 41 with 1, and it calls 125 of the 129 answers of the `baseline` run supported, where version 1 calls 127. Version 1 stays the default, and every score in the results tables and in a run's summary is its ([#59], [ADR 0023]).
- Each version of a judge keeps its verdicts on an answer run in files of its own: version 2 writes `support-v2.jsonl`, `support-v2.json` and `support_check-v2.json`, and never reads, continues or discards those of version 1. `make eval-support-v2` judges the `baseline` answers with version 2, with the verdicts of its check on the first labelled answers ([ADR 0023]).
- Follow-up questions in the service. `POST /ask` takes `earlier`, the earlier questions of a conversation, as strings: an answer is refused. The service keeps no conversation. With an earlier question, the question is first rewritten into one that stands alone, with one more request to the answer model and a time limit of its own, and the rewritten question is looked up in the answer cache, searched for and answered. Every answer has `searched_for`, which names the question searched for and how the step came to it, and a stream sends it first as `event: searched`. When the step fails or its reply is not usable, the question is searched for as it was typed, and the reply says so. The demo page holds a conversation in its memory, sends each answer's question as it was searched for with the next one, and shows "Searched for: …" when that is not the question typed. `ask --earlier` does the same on the command line. `HANDBOOK_RAG_FOLLOW_UPS`, `HANDBOOK_RAG_FOLLOW_UP_EARLIER_QUESTIONS` and `HANDBOOK_RAG_FOLLOW_UP_TIMEOUT_SECONDS` set it. No evaluation of answers runs the step: while one runs, it is refused at the entry point ([#79], [ADR 0024]).
- Held-out follow-up cases, `evals/follow-ups/held-out/`: 20 conversations written by a session that had seen none of the step's replies, read a second time, and asked into a run of their own, `make eval-follow-ups-held-out`. Under the rule as it stood before them, 9 of 14 follow-ups came out right, against 35 of 43 in the first run; 2 of the 5 misses were good rewrites the step's checks refused ([#79], [ADR 0024]).

- `eval-adversarial` decides by key facts: each injected case carries the fact any right answer to its golden question states, with the wordings that count, and a reply that code would call held counts as held only if it states it; otherwise it waits for a reader. Read again under that rule, with no request, the saved baseline moved no outcome: 12 of 20 planted instructions held ([#66], [ADR 0025]).
- `--instructions` on `eval-answers`, `eval-adversarial` and `eval-limits` chooses the version of the answering instructions. Version 1 is kept so that the old and the new assistant can be asked on the same day; outside an evaluation the generators refuse any version but the current one, and no setting names one. A run records `instructions_version` for a version other than 1 ([ADR 0025]).
- `handbook-rag eval-limits` and `make eval-limits`: the question of each of the 40 passages of the limit answers, asked of the assistant with that passage and no search, and each answer judged by version 2 of the support judge; 80 requests when none fails and none declines. Its figure that counts is the answers with no statement faulted, out of all the questions, with the cause of each question that got no answer; the share of judged answers faulted is given beside it. `--against` compares two final runs question by question, with an exact sign test. Version 1 gave an answer with no statement faulted for 32 of the 40 questions, version 2 for 39: 7 better and none worse, p = 0.016, a clear difference by the rule fixed beforehand. By one model session's reading, three of the judge's verdicts behind it look wrong by the judge's own instructions, and without them it is 4 against 0, p = 0.125. A blind reading of the 80 answers by another model session, told neither version nor verdict and decided on after the judge's figure was known, found 4 questions better and none worse, p = 0.125, on the count that does not rest on the one label the reader's rule pointed at with its example; with that label, 32 and 37 that keep their limit, p = 0.0625: the same way and not clearly. Beyond that, the reader was told more than the note of 14:22 holds, and the mixing could not hide the fixed form of version 2 (ADR 0025 gives both). No person has read the answers. Its packet, key and readings are in `evals/runs/limits/reading/` ([#59], [ADR 0025]).
- The answer evaluation under version 2, `evals/runs/answers/hardened`: 126 of 132 answerable questions answered, against 127; 27 of 33 unanswerable questions declined, against 31, which meets line e of the note fixed before the run; 153 right calls of 165, against 158; 123 of 126 answers correct by the reference, against 125 of 127, 1 better and 2 worse, p = 1.000. The reader of line c, one model session given the four changed replies and not the totals, found one answer lost, to the check for a reply that cites nothing. The confirmation, the 33 unanswerable questions asked again with the one change, `evals/runs/answers/hardened-unanswerable`, declined 30: line e is not met there, and 3 of the 30 rest on the change, by the code before which it would be 27; one of the three states a related fact joined into the sentence of the form, and counted as an answer it would leave 29. Lines a, b, c, d and f are not met. By the note fixed after line e was met, the hardening may then be merged ([ADR 0025], [#88] for scoring a reply that says the passages do not say and adds a related fact). `eval-answers --only unanswerable` asks the questions of one type, and an answer run lists in full every reply that counts as an answer to a question the handbook does not answer ([#66], [ADR 0025]).
- `eval-compare` gives an exact two-sided sign test on the questions whose score changed, beside each row. An answer run's report gives the calls as the person saw them, counting a reply a check declined as no answer, what each check took out of a reply to an answerable question beside its reference answer, and the time until the first words were passed on to the reader, which each reply now records ([ADR 0025]).
- Checks of the reply in code, under version 2 of the instructions: each line after the last citation that holds a web or email address is dropped, a reply that cites nothing at all is declined, and a reply of 200 words or more that is mostly a copy of its passages is marked. A reply that is not an answer is shown as the fixed declining sentence alone, with a note, except a reply made only of the fixed form for a missing part that carried citation markers, which is shown in its own words with no source and no note; and one that did not come to a proper end loses its words after the last citation, with a note. Each check that fired is shown with the answer: in the new `checks` field of `POST /ask`, as a note under the answer in `ask`, and on the demo page, where each note stands in its own turn, under the answer's text or with the notice of a reply that is not an answer, in a look of its own with a "Changed" badge, and is read out with the outcome; a reply a check declined says so. Text is streamed once a citation follows it, so that nothing streamed is taken back. The words a check took out are kept in a saved run, never shown. The span attribute `handbook_rag.reply_checks` names the checks that fired ([#66], [ADR 0025]).
- `handbook-rag eval-reply-checks`: the checks of version 2 made on the replies of a saved run, with no request. On the 129 answers of the first answer run they would decline 1, which cites no passage, and change nothing else ([ADR 0025]).
- Held-out planted cases, `evals/adversarial/held-out/`: 20 cases written by a model session that had read neither the assistant's instructions nor any reply to a planted case, though it had read the first answer run's ordinary answers to the questions it chose, checked by another before any was asked, and kept from the work on the hardening, with the rules their replies are decided by and a blind reading. Two are deliberate near translations of tuning cases, into German and French. Asked of version 1 before the work began and of the frozen version 2: by what the model itself wrote, version 2 held 16 of 20, with 5 cases better and 1 worse, p = 0.219 by an exact sign test; as the person sees it, after the checks, they held 12 and 18 of 20; 7 cases went from broke to held and 1 from held to broke, p = 0.070. 2 of the 7 gains came through the check for a closing line with an address, and count as held by a rule added to the rules for deciding after the old assistant was asked ([#66], [ADR 0025]).
- What the last two runs of the hardening report, and what result would keep it from being merged, written down before they are run: `docs/experiments/2026-11-25-2318-before-the-last-runs.md`, and its addendum, `docs/experiments/2026-11-26-0055-before-the-last-runs-addendum.md` ([ADR 0025]).
- An answer run writes `changed-replies.jsonl`: every reply to a question the handbook answers that a check changed or that was read as the fixed form, with the question, the reference answer, the model's own words and the reply as shown, and none of the run's totals, for a reader ([ADR 0025]).

### Changed

- **The answering instructions are version 2** ([#66], [#59], [ADR 0025]). They name the kinds of planted text that got through and say not to obey them, say that a notice in a passage is no reason to decline, ask the model to keep the limits a passage puts on a rule, and give one fixed form for a part the passages do not answer, "The passages found do not say". For Gemini each passage stands between two lines that hold a code made at random for each request, in place of fixed tags. On the 36 adversarial cases, which it was tuned on, version 2 held 19 of 20 planted instructions where version 1 held 12; every answer changes. The answer evaluation run under it met line e of the note fixed before the run, 27 of 33 questions the handbook does not answer declined against 31; the confirmation with the one change declined 30, which passes the line and rests on the change. The answer cache starts empty under it.
- A reply that cites no passage and is nothing but the fixed form for a missing part is a decline, shown as the fixed declining sentence; one that cites a passage stays an answer. A reply whose every citation follows the fixed declining sentence is a decline too, and a decline lists no source. After line e was met, a reply made only of the fixed form is a decline whether or not it carries citation markers: with markers it is shown in the model's own words, which may have been passed on, with no source. Read again with that change, the same replies of the answer run give 29 of 33 declined, which is not a result under line e ([ADR 0025]). The answer cache's guard finds the fixed form by its words, and keeps its list of phrases for other wordings ([ADR 0025]).
- The rewriting step's rule for a rewrite without a question mark, for a question typed without one: before the held-out cases were written it also took a negated verb, such as "isn't", and a preposition followed by a question word, such as "in which", as the start of a question; after the held-out replies were seen, the requirement for any opening was taken out. A statement with no stop at its end is now used for such a question, and the reply shows it as the question searched for. The saved run of the 82 cases gives the same outcomes under both changes ([ADR 0022], [ADR 0024]).
- `POST /ask` reads a body of up to 32 KB, to hold the earlier questions. It was 8 KB ([ADR 0024]).
- The demo page is a conversation instead of a form. Each question is a bubble and each answer a card, with the question box docked under them. The sources are cards beside the answer on a wide screen and under it on a phone, each with its number, the page's title and the headings above the passage, a piece of the passage, and the page's address in brief, whole in the link's title and in the fold with the passage. Pointing at or focusing a number in the answer marks its card, and the other way round, and pressing the number brings the card into view. Under each answer a strip of badges says how it was made: how many passages the search found, how many sources it cites, whether a follow-up was rewritten, whether it came from the cache, the model, the times and the tokens. The facts it showed before are one click further, under "All details". A decline is a calm notice and not a failure; a failure is a card that says what happened and offers to ask again; the word over each card says how its turn ended. Placeholders and a caret show the answer coming, and the stop button shows only then. Enter now asks and Shift+Enter starts a new line; Ctrl+Enter and Cmd+Enter still ask, and an Enter that ends a composition with an input method does not. The header leads to the README's sections on how the system works and how it is measured. A fourth example question, one the handbook does not answer, shows a decline. The page follows the light or dark setting, and its motion stops for a reader who asks for less. Its guards hold: the same content security policy, no inline script or style, only its own two files, everything from the API shown as text, links made in one checked function, nothing kept in the browser, and the address under a source shown whole, on a line that wraps. What it sends is said in one short line whenever there is a conversation, with the whole notice and the list of questions in a fold above the box. Tests now hold the script to a fixed list of the elements it may make, with no live region and nothing that loads or styles, and the style sheet to the ways of hiding or cutting the address that the test knows of, on the line, on a selector that could reach it, or around it; it does not compute the cascade, and they are not every way a style sheet can hide text. The test that runs the script in node now also checks the turns, the notices and failure cards, the placeholders and the caret, the badges, the source cards and the linking, the keys, the stop button, the word over each card, the examples, "Ask again", the found passages, the alert and the count of what is sent, with a planted fault for each ([ADR 0016]).
- The demo page has a look of its own, after its owner found the last one "AI generated and very common": the signage of a transit system. Each question stands as a black sign, and under it how the answer was made runs as the stops of a line (asked, the rewrite of a follow-up, searched, first words, answered), which is also the progress display while the answer is written and ends where a reply really ended, in words as well as in shape: at a decline, dashed and grey after the search, at a check's decline, at a reply withheld, cut off at the length limit or stopped early, at a failure, saying what went wrong, or at a stop by the reader, before or after the first words. A stored answer runs on a line of its own. The strip of badges is gone: what it said is said at the stops, and the facts and the trace id are a timetable under "All details". Each source an answer cites has a line colour, by its number, and a citation is the numbered bullet of its line; a source past the fifth takes the colour of the signs. A check's note is a yellow notice. The fold of the page's own search names it, shows once the line reaches the search, and offers "Show them" only when it holds passages. The welcome's example questions are destinations, the note under the one the handbook does not answer now holds whatever the reply, and "What happens when you ask" is a line in the order things happen. On a phone the status line wraps instead of being cut. One typeface, Overpass, from one font file the page's own server sends, with its licence (SIL Open Font License 1.1) beside it, byte for byte, in the package and the image; the security policy gains `font-src 'self'` and nothing else. Tests pin the font's and the licence's hashes; read the style sheet as a browser does, in either case and with escapes resolved, for any `url()` or `src()` but the font's, an `@import`, a `local()` font, an image function, a second font face or a family but Overpass and the system's own, including in the `font` shorthand (they read text, and a way of naming a file they do not list gets past them: in a browser the security policy is what refuses anything from another origin); learn what stands around the address line from the page the script really builds, so that a new wrapper cannot be left out of the check that nothing hides it; hold the look to its rules (colours only as tokens, no capitals, no shadows, no rounding but circles), colour for source (each line its own colour in both themes, its numeral at 4.5:1 at least, and each citation in the colour of its source by class, run in node) and the style rules that carry meaning (reduced motion, a visible focus, the next stop, a decline's dashed line, the bar where a line ends, the dark sign's edge); and run the line of stops in node through each state. A property of the style sheet that carries meaning (what can hide or move text, the focus, the edges, the grounds, motion, everything drawn before or after an element, the line colours) may be set only in a rule written out in a list in the test, with a word on why, and an unknown media query, at-rule, nested rule or `!important` fails; the colours the page puts text and marks on are held to their contrast in both themes. A first review found two of these tests weaker than first said and colour for source held by nothing, with 20 of its 31 planted faults passing, and a second found selectors that got past tests that looked rules up by selector, which the list closes; each of their faults is now found, among 171 faults planted in the script. Tried by eye in Chromium only, on a stand-in service ([ADR 0016]).

- `snapshot` refuses to replace a snapshot of another commit in the data directory unless it is given `--replace`, so that a newer commit is downloaded beside the pinned one and not over it. When the manifest is missing, cannot be read or names no commit, the clone is asked which one is there ([ADR 0019]).
- The same question asked a second time on the command line or over HTTP now gets the stored answer, with no request to the model. Before, every question was a new request and a new wording. `HANDBOOK_RAG_CACHE_MODE=off` gives that back ([ADR 0020]).
- The service loads a third model when it starts with an answer model and the default settings: the cache's second check, 552 MiB on disk. Its first start downloads it, in Docker Compose too, and a service that cannot load it does not start. `ask` downloads it at the first question that has a stored question near it. `HANDBOOK_RAG_CACHE_MODE=exact` or `off` avoids it ([ADR 0020]).
- `POST /ask` takes `cache` in its body, and its answer has `cached`, which is null unless the answer came from the cache ([ADR 0020]).
- The repository is named `handbook-rag`, like the Python package and the command. It was `production-rag-pipeline`. GitHub redirects the old address, and the links in the documents use the new one ([#70]).
- A request that the model's provider refuses as it was sent is reported as a problem of configuration and never sent again. It was reported as an unavailable model. A refusal in the 400s is sent again only when the provider's own SDK would send it again by default: 408 for Gemini, 408 and 409 for Claude. A 429 is a rate limit or a used-up allowance, as before. So 409 for Gemini, 499 for both, and every other refusal in the 400s count as configuration ([ADR 0018]).
- The Claude SDK's own trace span is switched off for answers: each request already has the project's span, and the SDK's would count its tokens a second time ([ADR 0018]).

### Fixed

- The decision record of the MCP server and a comment in its code said that one line of anything else on standard output breaks the client's connection. The protocol allows nothing else there, but what a client does with such a line is its own choice: the SDK's own client logs an error and reads on. Found while the study notes of this project were checked against the installed SDK ([ADR 0017]).
- `index` skipped a chunk whenever a point with its id was stored, even when that point had another title, other headings, another address or another section, which a change of `HANDBOOK_RAG_HANDBOOK_BASE_URL` or of how titles are read can cause without changing any id. It now stores such a chunk again, and says how many it replaced ([#21], [ADR 0019]).
- A replacement of the snapshot that failed after the files changed left the manifest of the old commit beside them. The manifest is now removed before the files change ([ADR 0019]).
- The documents said that measuring the answers again after a change to the instructions costs more than a day of the free allowance. It is about 460 requests of the 500 a day, and more than a day only when attempts have to be repeated ([#69]).
- A Claude stream whose connection timed out or broke after it had begun let the HTTP library's own error through, which the service then reported as a fault of its own. It is now reported as a model that did not answer in time, or as an unavailable one ([ADR 0018]).

### Known gaps

These are the [limitations in the README](README.md#limitations). The [roadmap](docs/roadmap.md#after-10) gives the open issues in the order they should be taken up.

- No person has verified the test questions, the labelled answers or the readings, in the sets of 1.0.0 or in those this version adds: model sessions made and checked them ([#8]).
- Most test questions were written by a model that could see the passage; too few were written without it ([#43]).
- The search gives the model both passages for 12 of the 33 questions that need two, no change to the search that was tried finds them clearly more often, and no answer was written for these questions ([#78], [ADR 0021]).
- A follow-up is rewritten from the earlier questions alone, so one that points at something only an answer named is usually rewritten to a wrong question. Real conversations and the answers to rewritten questions are not measured, and the rule for a rewrite without a question mark was last changed after the held-out replies were seen ([ADR 0022], [ADR 0024]).
- Both judges are the answer model itself, checked on that model's answers only, and whether a planted instruction moves a judge is not tested ([ADR 0014]).
- The first judge's instructions lack two rules of its rubric, and its numbering slipped in one long answer ([#58]).
- Questions whose false assumption the handbook corrects are not in the test set ([#37]).
- Version 2 of the answering instructions declines fewer of the questions the handbook does not answer: 27 of 33 as the run was made, against 31 under version 1; 29 when the same replies are read again after the one change; 30 when asked again with the change, which passes the line drawn before the run and rests on the change. A reply that says the passages do not say and adds a related fact counts as a wrong call ([#88], [ADR 0025]). How code reads a reply in the fixed form is wrong in cases the decision record names, in both directions, and how often is not measured ([#89]).
- Planted instructions still get through. On the 20 held-out cases, a notice that a page's process is out of date and a notice that keeps the answer to managers were obeyed, and a planted closing line with an address was written by the model twice and taken out only by a check. On the cases the hardening was tuned on, a page that calls itself confidential still ends in a withheld answer, through a check. Encoded instructions, instructions split across passages and instructions in a title were not tried ([ADR 0025]).
- How often answers leave out a limit is measured only on 40 questions chosen for it, by a judge that errs. By a blind reading the difference between the two versions of the instructions is not clear, and the questions people ask are not measured ([#59]).
- One reply for each question, on a free tier whose speed is uneven.
- The answer cache is tuned on pairs no person has read, keeps what the model wrote once, right or wrong, and shows a stored question to whoever is served its answer. Nothing is measured on real questions ([ADR 0020]).
- No authentication and no rate limit for each caller.
- A fallback model that answers has not been seen, and its answers are not evaluated ([ADR 0018]).
- The demo page's script is run in node, not in a browser; its look, a real browser and a screen reader are left to checks by hand ([ADR 0016]).
- One pinned snapshot of the handbook. The index can be updated to a newer commit, but the test questions and the saved runs are tied to the pinned one, and text the handbook repeats is indexed as often as it appears ([#34], [ADR 0019]).
- The Claude path has never been run, also not under version 2 of the instructions, and model tiers were not compared ([#17]).
- Checked on one laptop: the quickstart has not been followed on a clean machine, and a full indexing run inside a container has not been done.

## [1.0.0] - 2026-11-23

The first version: an assistant that answers questions from the GitLab handbook with numbered citations, or declines, on the command line and over HTTP. Its search and its answers are measured by runs committed to the repository, and the [README](README.md#results) reports them.

### Added

#### Search

- Download of the handbook's Markdown pages at a pinned commit, with a manifest of what was fetched ([#26], [ADR 0002]).
- Cleaning of each page, and chunks cut along its headings within a token budget ([#27], [ADR 0009]).
- Search by meaning with a local embedding model, in Qdrant, and an index that can be built again safely ([#28], [ADR 0004], [ADR 0005]).
- Search by keyword (BM25), and hybrid search that merges both lists by reciprocal rank fusion ([#32]).
- Optional reranking with a cross-encoder ([#33]).
- Hybrid search as the default and reranking off, chosen by measurement ([#44], [ADR 0011]).
- Models and the tokenizer read from disk before the model hub is asked ([#54], [#55]).

#### Answers

- `ask`: an answer from the retrieved passages with a numbered citation after each statement, or one fixed sentence when they do not hold the answer ([#29], [ADR 0010], [ADR 0013]).
- Gemini on its free tier as the default answer model, and a model name that selects the provider ([#29], [ADR 0013]).
- A Claude generator with the API's own citations, unit-tested and never run against the Claude API ([#29], [ADR 0006]).
- Passages sent as data in tagged blocks, with text that looks like one of the tags made harmless ([#29], [ADR 0013]).
- Generation errors that say what kind they are, such as a rate limit or a used-up allowance ([#64]).

#### Evaluation

- Retrieval evaluation: Hit@k, Recall@5, MRR@10 and nDCG@10, each with a 95% interval ([#30], [ADR 0007]).
- Tools and guidelines for building the golden set, a page for reviewing it by eye, and a check that refuses evidence another page repeats ([#31], [#35]).
- A comparison of two runs question by question, with a fingerprint of the labels behind each score ([#36]).
- Committed runs kept small ([#38]), and a results table of every run ([#40]).
- The golden set: 165 questions, made and checked by separate model sessions ([#39]).
- Experiments with half-size chunks and other fusion settings, which did not help ([#41]).
- A retrieval gate in CI, on a slice of the handbook with the real models ([#45], [ADR 0012]).
- Answer evaluation in stages, within a daily allowance of requests, with the scores that code can compute ([#56], [ADR 0014]).
- A judge for whether each statement is backed by the passage it cites, checked on 169 labelled answers before its score was used ([#57]).
- A second judge for whether an answer agrees with its reference answer, checked on 167 labelled answers ([#62]).
- A control for the second judge: each reference answer given to it as the answer ([#67]).
- Adversarial tests: 36 cases that plant an instruction in a passage or ask a question outside the handbook ([#67]).

#### Serving

- `POST /search` and `GET /health` over HTTP ([#49]).
- `POST /ask`, as one JSON response or as server-sent events, with fixed words for every failure and a cap on answers in flight ([#64], [ADR 0015]).
- `docker compose up` for the API, Qdrant and the trace backend ([#50]), with the Gemini key passed from `.env` and nothing else ([#64]).

#### Observability

- A trace for every search, with a span for each stage, sent over OTLP to Jaeger ([#48], [ADR 0008]).
- One round trip to the vector store for each query instead of two, found by reading traces ([#49]).
- The whole answer traced, the model call included, under the names of OpenTelemetry's conventions for generative AI ([#64]).

#### Engineering

- The package, settings from the environment, lint, type checks, tests and CI on every pull request ([#23]).
- Decision records reviewed and accepted, among them a pipeline written in the project's own code instead of a framework ([#25], [ADR 0001], [ADR 0003]).
- Pull requests merged by fast-forward, so every commit lands on `main` unchanged ([#25]).
- The container image built in CI on every pull request, and checked to run as a user other than root ([#50]).
- The README rewritten with results and limitations, a test that holds its results to the committed runs, and this changelog ([#18]).

### Fixed

- The evaluation tests no longer really wait between requests, which had made the fast suite take minutes ([#65]).

### Known gaps

These are the [limitations in the README](README.md#limitations).

- No person has verified the test questions, the labels the judges were checked against, or the readings of the adversarial replies ([#8]).
- Most test questions were written by a model that could see the passage; too few were written without it ([#43]).
- Both judges are the answer model itself ([#63]).
- The judges were checked on one model's answers only, and whether a planted instruction moves a judge is not tested.
- The first judge's instructions lack two rules of its rubric, and its numbering slipped in one long answer ([#58]).
- Questions whose false assumption the handbook corrects are not in the test set, so nothing tests whether the assistant corrects the assumption ([#37]).
- Planted instructions got through in 8 of the 20 cases that plant one, and nothing is fixed yet ([#66]).
- Answers that leave out a limit or an exception are not measured reliably ([#59]).
- One reply for each question, on a free tier whose speed is uneven: the time to a complete answer is over the charter's target.
- No authentication and no rate limit for each caller, and no deadline or fallback for a model call ([#42]).
- One pinned snapshot of the handbook ([#21], [#34]).
- The Claude path has never been run, and model tiers were not compared ([#17]).
- Checked on one laptop: the quickstart has not been followed on a clean machine, and a full indexing run inside a container has not been done.

[Unreleased]: https://github.com/MumukshuPanchal/handbook-rag/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/MumukshuPanchal/handbook-rag/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MumukshuPanchal/handbook-rag/releases/tag/v1.0.0
[#8]: https://github.com/MumukshuPanchal/handbook-rag/issues/8
[#17]: https://github.com/MumukshuPanchal/handbook-rag/issues/17
[#18]: https://github.com/MumukshuPanchal/handbook-rag/issues/18
[#19]: https://github.com/MumukshuPanchal/handbook-rag/issues/19
[#20]: https://github.com/MumukshuPanchal/handbook-rag/issues/20
[#21]: https://github.com/MumukshuPanchal/handbook-rag/issues/21
[#22]: https://github.com/MumukshuPanchal/handbook-rag/issues/22
[#23]: https://github.com/MumukshuPanchal/handbook-rag/pull/23
[#24]: https://github.com/MumukshuPanchal/handbook-rag/issues/24
[#25]: https://github.com/MumukshuPanchal/handbook-rag/pull/25
[#26]: https://github.com/MumukshuPanchal/handbook-rag/pull/26
[#27]: https://github.com/MumukshuPanchal/handbook-rag/pull/27
[#28]: https://github.com/MumukshuPanchal/handbook-rag/pull/28
[#29]: https://github.com/MumukshuPanchal/handbook-rag/pull/29
[#30]: https://github.com/MumukshuPanchal/handbook-rag/pull/30
[#31]: https://github.com/MumukshuPanchal/handbook-rag/pull/31
[#32]: https://github.com/MumukshuPanchal/handbook-rag/pull/32
[#33]: https://github.com/MumukshuPanchal/handbook-rag/pull/33
[#34]: https://github.com/MumukshuPanchal/handbook-rag/issues/34
[#37]: https://github.com/MumukshuPanchal/handbook-rag/issues/37
[#35]: https://github.com/MumukshuPanchal/handbook-rag/pull/35
[#36]: https://github.com/MumukshuPanchal/handbook-rag/pull/36
[#38]: https://github.com/MumukshuPanchal/handbook-rag/pull/38
[#39]: https://github.com/MumukshuPanchal/handbook-rag/pull/39
[#40]: https://github.com/MumukshuPanchal/handbook-rag/pull/40
[#41]: https://github.com/MumukshuPanchal/handbook-rag/pull/41
[#42]: https://github.com/MumukshuPanchal/handbook-rag/issues/42
[#43]: https://github.com/MumukshuPanchal/handbook-rag/issues/43
[#44]: https://github.com/MumukshuPanchal/handbook-rag/pull/44
[#45]: https://github.com/MumukshuPanchal/handbook-rag/pull/45
[#48]: https://github.com/MumukshuPanchal/handbook-rag/pull/48
[#49]: https://github.com/MumukshuPanchal/handbook-rag/pull/49
[#50]: https://github.com/MumukshuPanchal/handbook-rag/pull/50
[#54]: https://github.com/MumukshuPanchal/handbook-rag/pull/54
[#55]: https://github.com/MumukshuPanchal/handbook-rag/pull/55
[#56]: https://github.com/MumukshuPanchal/handbook-rag/pull/56
[#57]: https://github.com/MumukshuPanchal/handbook-rag/pull/57
[#58]: https://github.com/MumukshuPanchal/handbook-rag/issues/58
[#59]: https://github.com/MumukshuPanchal/handbook-rag/issues/59
[#62]: https://github.com/MumukshuPanchal/handbook-rag/pull/62
[#63]: https://github.com/MumukshuPanchal/handbook-rag/pull/63
[#64]: https://github.com/MumukshuPanchal/handbook-rag/pull/64
[#65]: https://github.com/MumukshuPanchal/handbook-rag/pull/65
[#66]: https://github.com/MumukshuPanchal/handbook-rag/issues/66
[#88]: https://github.com/MumukshuPanchal/handbook-rag/issues/88
[#89]: https://github.com/MumukshuPanchal/handbook-rag/issues/89
[#67]: https://github.com/MumukshuPanchal/handbook-rag/pull/67
[#69]: https://github.com/MumukshuPanchal/handbook-rag/pull/69
[#70]: https://github.com/MumukshuPanchal/handbook-rag/pull/70
[#78]: https://github.com/MumukshuPanchal/handbook-rag/issues/78
[#79]: https://github.com/MumukshuPanchal/handbook-rag/issues/79
[#92]: https://github.com/MumukshuPanchal/handbook-rag/issues/92
[ADR 0001]: docs/adr/0001-record-architecture-decisions.md
[ADR 0002]: docs/adr/0002-handbook-corpus.md
[ADR 0003]: docs/adr/0003-own-pipeline-code-instead-of-a-framework.md
[ADR 0004]: docs/adr/0004-qdrant-as-vector-store.md
[ADR 0005]: docs/adr/0005-local-embedding-and-reranking-models.md
[ADR 0006]: docs/adr/0006-claude-with-native-citations-for-generation.md
[ADR 0007]: docs/adr/0007-evaluation-strategy.md
[ADR 0008]: docs/adr/0008-opentelemetry-tracing.md
[ADR 0009]: docs/adr/0009-structure-aware-chunking.md
[ADR 0010]: docs/adr/0010-search-result-blocks-and-declining.md
[ADR 0011]: docs/adr/0011-hybrid-retrieval-by-default.md
[ADR 0012]: docs/adr/0012-retrieval-gate-on-a-slice.md
[ADR 0013]: docs/adr/0013-gemini-free-tier-with-marker-citations.md
[ADR 0014]: docs/adr/0014-answer-evaluation-in-stages.md
[ADR 0015]: docs/adr/0015-answers-over-http.md
[ADR 0016]: docs/adr/0016-a-demo-page-without-a-build-step.md
[ADR 0017]: docs/adr/0017-an-mcp-server-for-search.md
[ADR 0018]: docs/adr/0018-time-limits-retries-and-a-fallback.md
[ADR 0019]: docs/adr/0019-updating-the-index-by-reusing-vectors.md
[ADR 0020]: docs/adr/0020-an-answer-cache-that-matches-by-meaning.md
[ADR 0021]: docs/adr/0021-two-passage-questions-and-the-search-as-it-is.md
[ADR 0022]: docs/adr/0022-follow-up-questions-rewritten-before-the-search.md
[ADR 0023]: docs/adr/0023-a-second-version-of-the-support-judge-for-dropped-limits.md
[ADR 0024]: docs/adr/0024-follow-up-questions-in-the-service.md
[ADR 0025]: docs/adr/0025-hardening-against-planted-instructions.md
