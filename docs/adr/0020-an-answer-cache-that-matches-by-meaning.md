# ADR 0020: An answer cache that matches by meaning

- Status: Accepted
- Date: 2026-11-24

## Context

Every answer costs one request to the answer model, and sometimes more. On the free key that is part of an allowance of 15 requests a minute and 500 a day, and a wait: a model call took a median of 2.4 seconds in the answer evaluation, and 10.1 seconds at the 95th percentile ([the run](../../evals/runs/answers/baseline)). A question that was answered before should not cost that again ([issue 24](https://github.com/MumukshuPanchal/handbook-rag/issues/24)).

The simplest cache serves a stored answer to the very same question. People do not ask the very same question. So the question was whether a cache may also serve a stored answer to a question that *means* the same, and what that risks.

**The owner of the project decided that it may.** The reason, in the owner's words: "in production we need to take care of cost and latency and every user has their own vocabulary to ask questions". This record does not argue that decision. It records it, puts the measured risk beside it, and says what was built to reduce the risk and to make it visible.

**The risk was measured before anything was built** ([the evaluation guide](../evaluation.md#a-semantic-cache-measured-before-it-is-built), run [`baseline`](../../evals/runs/cache/baseline)). On 198 labelled pairs of a test question and either a rewording or a near-miss:

- **The cosine of two question embeddings does not tell a rewording from a near-miss.** No threshold served a rewording without serving a near-miss.
- **A second model does far better, and is not always right.** It reads the two questions together and says whether each implies the other. At a threshold chosen on one half of the pairs so that nothing wrong was served there, it served 7 of 49 rewordings on the other half and none of 50 near-misses. That half held no hard near-miss, and on 97 of 200 other splits of the same pairs something wrong was served.
- **The two near-misses it scored highest were both a stored decline.** The stored question was one the handbook does not answer, and the question asked was one it does. A cache would have told the asker that the handbook lacks what it holds.

The pairs were written and labelled by model sessions. No person has read them.

## Decision

An answer cache with two tiers, on by default in the mode `semantic`.

**What goes wrong, said once.** A question that only looks like a stored one will sometimes be served the stored question's answer. Everything below reduces how often, or makes it visible when it happens. Nothing below ends it.

### Two tiers, in this order

1. **Exact.** The question is made comparable and looked up by a key. A match is served whatever it holds, a decline too: the same question has the same right reply. Comparable means, in this order: Unicode's composed form; every run of white space becomes one space, with none at the ends; case does not count; question marks, exclamation marks, full stops and spaces at the end are dropped. Nothing else is changed, so "What is C#?" and "What is C?" stay two questions. Case can carry meaning, as in "US" and "us", and this tier does not see it.
2. **By meaning.** The question is embedded as the search embeds it. The stored questions nearest to it by cosine are candidates: at most 3, none below a cosine of 0.7. The second model, the verifier, reads each candidate with the new question in both directions. The candidates are tried nearest first, and the first whose weaker direction reaches the threshold is served.

### What is kept from the second tier

- **A stored decline.** This is the guard the first measurement asked for. It was chosen after that run had been read, with every pair in view.
- **An answer in which one of two guards finds that part of the question was declined.** The part the passages did not cover may be the very thing a neighbouring question asks. The first guard looks for the fixed sentence of a decline beside an answer. The second looks for a phrase by which a model says in words of its own that something is not in the handbook, such as "not provided", "does not cover" and "do not specify" (`MISSING_PHRASES` in `src/handbook_rag/cache/rules.py`). The instructions ask the model to "say plainly what is not covered" and fix no words for it, so the second guard can only be a list of phrases. It holds the four phrases the model used in the saved answer run and their relatives: other verbs of saying, the instructions' own "do not contain", the shortened forms, and each with one of a few adverbs in between. **It errs towards keeping an answer out**: "dental care is not covered" is a plain answer about a benefit, and it matches. **And a reply can get past it**: a gap worded in a way the list does not hold is not recognised, and that answer is a candidate like any other. **The list is fitted to four replies of one answer run**, and it reads English only. On 5 replies it was not written from, which a model session wrote for [the second judge's labelled answers](../../evals/judge/README.md) to give none of what was asked, it finds 1.
- **An answer that cites no passage.** It rests on nothing a reader can check.
- **An answer written from another part of the handbook.** A question asked of one section never gets an answer written from another section or from all of it, in either tier.

Only what these recognise is kept out. Each of the first three is still served to the very same question, by the exact tier. They are left out inside the search, so that none of them takes a nearer candidate's place.

On the 165 replies of the saved [answer run](../../evals/runs/answers/baseline), the cache's own code keeps 41 from the second tier: 36 declines, 4 answers by the second guard, and 1 that cites no passage. The first guard finds none. Of the 4, 2 are answers to questions that the handbook answers, which is the guard erring as said ([the guide](../evaluation.md#the-answer-cache-measured-as-it-is-built) names each reply).

### What is stored, and when

Only a whole reply written by the answer model. Not stored:

- a failure, or a reply that broke off: nothing is kept of either;
- a reply cut off at the length limit, withheld by the provider, or stopped for another reason;
- a reply the fallback model wrote, because its answers are not evaluated ([ADR 0018](0018-time-limits-retries-and-a-fallback.md));
- a decline made without asking a model, because the search found nothing. That happens while an index is empty or half built, and must not outlive it;
- a reply with no text.

A decline the model wrote is a whole reply, and is stored.

Storing never fails an answer and never delays one. The answer goes out first, and a thread of its own writes it to the store. If the store is down, the failure is logged and nothing else happens. An answer that is still waiting to be stored when the process ends is lost, and that is logged too.

**A lookup waits 2 seconds at most.** After that it is given up, the failure is logged, and the question is answered as if there were no cache. A store that cannot be read, and a verifier that fails, count as a miss too. Without the limit, the store's own time limit would stand before every answer: 10 seconds in the service and 60 on the command line. The limit is a constant in the code, `LOOKUP_SECONDS`, and no setting. In the first check with the real service the slowest lookup took 1.2 seconds.

**What is slow only the first time in a process is not counted**: loading the verifier, its first reading, and the first embedding. A service does all three when it starts. A command is a new process for each question, so its one lookup would pay for them inside the limit. In three fresh processes on the development laptop the first embedding took 138 to 176 ms and the verifier's first reading 218 to 670 ms, where the second took 24 to 122 ms and 61 to 166 ms. A review of this change saw the first lookup of a fresh process given up at the limit, and the command then spends a request it need not. A longer limit for the command was the other way to mend that. It would make a command wait longer for a store that hangs, and it would still be a guess at how slow a first run can be. Not counting the first run asks no model anything twice, and an exact match, which runs no model, pays nothing for it.

**A lookup that was given up starts nothing more.** The verifier reads for one lookup at a time, and each waits its turn. A lookup whose time runs out while it waits gives its turn up, so a reading that nobody waits for does not keep the next question waiting. A reading that has begun is finished, and what it finds is thrown away.

### Where it is kept, and why entries have no time limit

In Qdrant, in a collection of its own beside the handbook's, so nothing new has to be run. A point is one answer: its vector is the embedding of its question, and its payload is the answer with the passages it cites, whole.

**The collection is made before the first question is looked up**: by the service when it starts, and by `ask` before it looks its question up. When the collection is there, that is one question to the store. A service that starts while its store is down logs that and makes the collection with the first answer it stores. For those moments a lookup finds nothing or fails, and either is a miss. Emptying the cache removes the answers and keeps the collection, so a service that runs under the same settings does not meet a collection that is being made again. `cache-clear --all` removes the collections of other settings whole. A service that still runs under one of those makes its collection again with the next answer it stores.

**The collection's name is a hash of every setting that decides what the right answer is.** When one of them changes, the cache has another name, and the answers stored before are not served under the new settings. They stay in their collection. They are served again when the setting is put back, unless `cache-clear --all` removed them or `index` emptied them in between. The name covers:

- the pages: the repository, the commit, the directory read, the address the sources link to, and the size below which a page is skipped;
- the index: the embedding model, the keyword model and the chunking;
- the search: its mode, and what that mode uses of the merging and reranking settings, the instruction a question is embedded with, and how many passages the model is given;
- the answer: the answer model, its length limit, for a Claude model its effort, the instructions, and the layout of a request, taken from a sample request the code builds;
- a version number in the cache's own code, for what no setting shows. The code that reads a model's reply, the code that finds passages and the cache's own rules decide an answer too. A person raises the number when one of them changes what an answer is.

A test lists every setting of the application as one that decides an answer or one that does not. A setting added later has to be put in one of the lists, or the test fails.

**So no time limit is needed.** The handbook is pinned to one commit, and nothing an answer depends on changes while the name stays the same. Two things can change under the same name, and each has its own remedy:

- **The index itself**, while it is first built, when a run that stopped is finished, or when pages are read another way. When a run of `handbook-rag index` has embedded, reused or removed a chunk, and has finished, it empties every cache collection of that handbook commit, under whatever settings each was made: one index serves many of them. A run that stops halfway has changed the index and emptied nothing. The run that finishes the work empties the caches if it still finds something to change. If it finds nothing, `handbook-rag cache-clear --all` is the remedy. An answer that a running service is still writing when `index` empties the caches is stored afterwards, and stays.
- **The code that decides an answer.** The version number is raised by hand. A test pins a hash of those modules, and of the functions that build a retriever and an answer model from the settings, and fails when one of them changes, which is the moment to ask. Whether the number has to go up is still a person's judgement.

A time limit would become necessary if the handbook were read live instead of from a pinned commit, if the answer model were named by an alias that moves, or if answers depended on the day they were asked. The settings refuse a model name that ends in `-latest`. They know no other alias.

### A cached answer says that it is one

The answer carries which tier served it, the stored question it was written for, the cosine, the verifier's score and when it was stored. It counts 0 requests to the model. `ask` prints this, `POST /ask` returns it as `cached`, also in the `answer` event of a stream, and the demo page shows "It was written for the question: …" above the answer, as text. For a match by meaning that is another wording than the asker's own, and reading it is how a wrong match is noticed.

**A caller can ask for a fresh answer**: `ask --no-cache`, or `"cache": false` in the body of `POST /ask`. Nothing is looked up, and the model is asked. If the new answer is one the cache keeps, it is stored under the question's own key, in place of what was there. The exact tier comes first, so that question gets its own answer from then on. A reply that is not kept, one cut off at the length limit for example, leaves the stored answer where it is.

### One place in the code, and never in an evaluation

`answer_question` takes the cache as an optional argument, absent by default. The `ask` command and the service pass one. **An evaluation never does**: it measures what the model writes, and a stored answer would be scored as if the model had just written it.

A switch holds this, for the whole process (`src/handbook_rag/measuring.py`). While it is on, building a cache fails, and so does handing one to `answer_question`, however the call is written. Every command that measures runs inside it: each `eval-…` and `golden-…` command, the judges' checks among them. So does every live test. One test lists each command of the application as one that measures or one that does not, so a new command has to be put in one of the lists, and runs each measuring command to see the switch on. A second test reads the source for every use of `answer_question`, as a second net.

### The settings, and where the defaults come from

| Setting | Default | What it is |
|---|---|---|
| `HANDBOOK_RAG_CACHE_MODE` | `semantic` | `off`, `exact`, or `semantic` for both tiers |
| `HANDBOOK_RAG_CACHE_THRESHOLD` | 0.91 | what the verifier's weaker direction has to reach |
| `HANDBOOK_RAG_CACHE_CANDIDATES` | 3 | how many of the nearest stored questions are checked |
| `HANDBOOK_RAG_CACHE_FLOOR` | 0.7 | the cosine below which a stored question is no candidate |

**The threshold is fitted on the labelled pairs, and nothing is known about how it behaves on real questions.** The measurement was run twice more after the first ([the guide](../evaluation.md#the-measurement-again-with-stored-declines-left-out) has both):

- **With stored declines left out by the golden set's label** ([`no-decline-candidates`](../../evals/runs/cache/no-decline-candidates)). That run takes a label's word for what a cache holds.
- **As the cache is built** ([`as-built`](../../evals/runs/cache/as-built)). The simulated cache is filled with the replies of the saved answer run, and the cache's own code decides which of them are candidates. The defaults rest on this run.

The rule is the first run's: on one half of the pairs, the threshold is put above the highest score at which something wrong is served, halfway to the next score. In both later runs that gives 0.8669 by the weaker direction. What is known of how that threshold behaves is what the third run shows:

- **On the half that chose nothing**, it served 16 of 49 rewordings right, and nothing wrong: none of 50 near-misses, and no rewording another stored question's answer.
- **That half did not test it hard.** No near-miss of that half would be served above 0.143. The hardest thing it holds is a rewording, `s073`, which would be served another stored question's answer at any threshold up to 0.854. Nobody has checked whether that other answer would be wrong for it: a label covers a pair's own question only, so the run counts it as wrong.
- **On 200 other splits of the same pairs**, the same rule served something wrong on the reporting half in 73, by the weaker direction. Those are 73 of the 111 splits that put `s073` on the reporting half.

**The default threshold is 0.91, and not the rule's 0.8669.** Where a threshold stands in its gap is a choice. The near-miss that set the rule's threshold scores 0.8206, the next score seen on that half is 0.9132, and the rule puts the threshold halfway. With every pair in view the gap is narrower: `s073` would be served the other answer up to 0.854. Every threshold from 0.8540 to 0.9131 serves each question of the third run exactly what 0.8669 serves it, and from 0.8662 the same holds for the second run. The default is the top of that range, to two places. It costs nothing on the runs: the half that chose nothing is served the same 16 of 49. And it stands 0.056 above the wrong serve at 0.854, where the rule's halfway stands 0.013 above it. That choice was made with every pair in view, so nothing tests it.

**The default floor is 0.7, where all three runs were made with 0.5.** It was read from the same pairs, with every pair in view. In each run no candidate under a cosine of 0.7612 scores above 0.13 by the weaker direction, and the lowest cosine of a rewording to its own stored question is 0.737. With a floor of 0.7 the rule gives the same thresholds on each run and no question is served anything else. The verifier then reads 234 candidates for the 198 questions of the third run, where it reads 592. **The floor does nothing for correctness**: the wrong serve at 0.854 is at a cosine of 0.789, above it. It saves the verifier's time on a question that has nothing near it. It can cost a saving: a rewording that lies further from its stored question than any in the set is no longer served, and the pairs cannot say how often that happens.

The number of candidates, 3, is the measurement's own setting. Nothing chose it from the pairs.

### What the default mode costs

A third model, the verifier, beside the embedding model and the keyword model. Measured on the development laptop on 24 November 2026 ([the guide](../evaluation.md#checked-with-the-real-service)):

- **On disk**: 552 MiB, downloaded the first time it is used.
- **In memory**: 310 MiB more at its peak, in a process that already held the embedding model.
- **To load**: 0.56 seconds, and 0.43 more for its first check. When the service started for the first check, loading it and asking it once took 1.1 seconds.
- **For each question that has a candidate**: a median of 32 ms for one candidate and 53 ms for three, each read in both directions.
- **For each question the cache cannot answer**: the lookup, before the search begins. In the first check with the real service the seven lookups that served nothing took between 95.0 and 268.9 ms. That check ran with a floor of 0.5, which kept little out: unrelated questions were candidates at a cosine of 0.518, and the verifier was asked about them. With the floor of 0.7 they are no candidates. What a miss takes since then was not timed with the real service.

**When it is loaded.** The service loads it when it starts and asks it once, so that no question waits for it. A service that cannot load it does not start, and the error names the two settings to change: `HANDBOOK_RAG_CACHE_VERIFIER_MODEL`, or `HANDBOOK_RAG_CACHE_MODE` for `exact` or `off`. A service without an answer model builds no cache and loads nothing. The `ask` command loads it only when a stored question is near the new one: a question with no candidate, and one that is matched exactly, never load it. In the modes `exact` and `off` it is never loaded.

**The first use downloads it.** For the service that is the first start. For `ask` it is the first question that has a stored question above the floor, and the command waits for the download before it answers. How long the download takes was not measured.

**In Docker Compose** the mode is written out in `compose.yaml` as `semantic`, the default. The first start of the container downloads the verifier with the other models into the volume for model files, and the container is given 600 seconds to start before its health check counts. Changing that line to `exact` or `off` avoids the download.

### Tracing, and two commands

The lookup is a span, `cache.lookup`, under `answer`: the tier, hit or miss, how many candidates, and for each candidate its cosine and the verifier's two directions. The stored questions are recorded only when `HANDBOOK_RAG_TRACE_CONTENT` is on. A hit ends the trace there: no search span and no model span follow ([the observability guide](../observability.md#the-trace-of-a-cached-answer)).

`handbook-rag cache-status` says how many answers are held, how many of them may be served by meaning, and how many were left behind under other settings. `handbook-rag cache-clear` empties the cache of the current settings and keeps its collection. With `--all` it also removes the collections left behind under other settings, whole. It touches only a collection whose name has the form this project gives: one that something else made is left alone, also when its name starts with `answers-`.

## What the checks with the real service showed

On 24 November 2026 the service was asked 13 times on the real index with the real key, and 7 of those cost a request to the model ([the guide](../evaluation.md#checked-with-the-real-service) has each one). It is one check by hand. It shows that the parts work together, and it gives no rate. It was made with the defaults of that day, a threshold of 0.8669 and a floor of 0.5, before the second guard and the third run. By its scores each question would be served or turned down the same way under the defaults of today. A second check, after the fixes of the review, is at the end of this section.

- **The same question again cost no request** and came back in 0.046 seconds, against 1.236 for the first. In other case and spacing: 0.013 seconds.
- **A rewording was served by meaning** in 0.166 seconds: a cosine of 0.929, and 0.998 by the verifier.
- **A near-miss was turned down**: FMLA leave against parental leave, a cosine of 0.896, and 0.042 by the verifier. The model was asked, and gave another answer.
- **A stored decline was served to its own question and to no other.** A rewording of it was answered by the model, which declined again. So was an answerable near-miss, the one of pair `d081`, which the verifier scored 0.947 in the first measurement: the stored decline was no candidate for it.
- **The model answered a question it should have declined, and the cache kept the answer.** Asked how much parental leave GitHub gives, it answered with GitLab's 16 weeks. In the answer evaluation it had declined the same question. A rewording of that question was not matched to it, by 0.005 in one direction, and got the same wrong answer from the model.
- **It missed a fault.** Its questions were asked seconds apart. Asked at once, in a check by hand during the review, a repeat was not served and a third lookup failed: the first answer to be stored was still making the collection in the background. That is why the collection is now made before the first question, and a slow test against a real store holds it.

**Checked again after the fixes**, on 24 November 2026, with the defaults of today, a threshold of 0.91 and a floor of 0.7. It is the account of the session that coordinated this change, of a single run on one laptop, and no file of it is in the repository. The service was started cold, with no cache collection in the store, and asked at once.

- **The collection was there before the first question**: it existed as soon as `/health` answered.
- **A first question** about a damaged laptop was answered by the model in 1.9 seconds, with 1 request.
- **The same question in other case and without its question mark** was served by the exact tier in 12.9 ms, with 0 requests.
- **A rewording** was served by meaning in 152 ms, with 0 requests: a cosine of 0.966, and 0.9945 by the verifier.
- **A near-miss**, about a damaged monitor, was not served. The model was asked, and declined after 11.7 seconds, with 1 request.
- **No lookup failed**: the log held no line that the cache could not be searched. The check cost 2 requests.

## Alternatives considered

- **Exact matches only.** Nothing wrong is served that the model did not write for that very question. It is what the mode `exact` gives. The owner decided against it as the default, for the reason given above.
- **The cosine alone, with a high threshold.** The usual semantic cache, and one model fewer. On the labelled pairs no threshold served a rewording without a near-miss, so it is not offered as a mode.
- **A time limit on entries.** Not needed while every setting an answer depends on is in the collection's name, with a version number for the code beside them. It would also remove right answers and cost the requests to write them again.
- **Storing declines for the second tier too.** A decline reused for a reworded question that the handbook does not answer would save a request. The measurement showed the opposite case, a decline served to a question that has an answer, to be the one the verifier is worst at.
- **The mean of the verifier's two directions.** On the third run it served 17 of 49 rewordings right on the reporting half against 16, and 1 more another stored question's answer. The measurement's rule took the weaker direction, and the cache follows the rule.
- **A model that judges whether a reply declines in part.** It would know a gap in any words. It is one more model call for every answer stored, on a key with a daily allowance, and a judge that has to be checked first as the two judges of the evaluation were. A list of phrases costs nothing and is plain about what it misses.
- **A cache in the process's memory, or a new store.** Lost when the process ends and not shared between the command and the service, or one more thing to run. Qdrant is already there and already searches vectors.
- **Checking candidates one by one, stopping at the first that passes.** It would save part of the verifier's work on a hit. One batch for all candidates is simpler, and three candidates took a median of 53 ms.

## What is still missing

- **An exact guard for a decline in part.** A fixed form for "this part is not in the handbook" in the answering instructions would let the second guard look for that form, and not for a list of phrases. It changes the instructions, so the answers would have to be measured again. It is planned with the next change of the instructions, and is not in this change. Until then the guard is a fit to four replies. That `d067` is not served in the third run is that fit, and no test of the guard.
- **A label for what the second guard keeps from `d067`.** The reply stored for `u004` names the quarters that `d067` asks about, so served to it, it might have been a right answer. The pair's label says `different`, and the rule counts by labels. Nobody has read that reply against that question.
- **A person's reading of the pairs.** The threshold is fitted on pairs that only model sessions wrote and labelled.
- **Any figure about real questions.** How often a near-miss arrives, how often a rewording does, and so how many requests the cache saves and how many wrong answers it serves: nothing here measures them. A production system would collect labels from real traffic. This project has no traffic.
- **A check of what is served.** A production system would answer a sample of cache hits again with the model, compare the two, and raise an alarm when they differ. That costs requests on a regular schedule, which a free key does not have to spare, and needs a judge for "the same answer", which would have to be checked first as the two judges were.
- **A limit on the cache's size.** Entries are never removed, except by `cache-clear`, by `index` when the index changes, or by being left behind when a setting changes. A production cache would remove the least used. Here a day's allowance is 500 requests, so the cache grows by 500 answers a day at most.
- **A larger cache than the measurement's.** The simulated cache held 165 replies, 124 of them candidates. The more a cache holds, the more stored questions sit near each new one, and the measurement says nothing about a cache of thousands.
- **Protection against a wrong answer that is stored.** The cache keeps what the model wrote once. The model does not write the same thing each time, and the first check above saw it answer a question it had declined before. Without a cache the next asker may get the right reply. With one, they get the stored one until someone asks with `--no-cache`.
- **One request for two people who ask the same thing at once.** Both miss, both ask the model, and the second answer replaces the first.
- **A hit without a free answer slot.** A cached answer is looked up in one of the service's four answer slots, so a full service turns away a question it could have answered from the cache.
- **Less start-up for the command.** `ask` loads the embedding model before it looks a question up, also when the exact tier then answers. On the laptop the whole command took 18.7 seconds for an exact hit, almost all of it start-up.
- **The embedding made once.** On a miss the question is embedded by the cache and again by the search, 13 ms at the median on the laptop.
- **The other order of a pair.** The measurement stores the test question and asks the other one. Labels that depend on which was asked first were not simulated the other way round.

## Consequences

- A repeated question costs no request and no wait for the model. A reworded one sometimes does not either. In the third run, on the half of the labelled pairs that chose nothing, 16 of 49 rewordings were served: 0.327, with a 95% interval of 0.212 to 0.466 that is too narrow, because pairs share test questions.
- A near-miss will sometimes get another question's answer, and the reader's only protection is the line that names the question the answer was written for.
- **The stored question is shown to whoever is served its answer.** A question can hold personal details, and another person then reads them. The service has no login and is for one machine. On a shared service this is a reason for the mode `exact`, where a reader is shown only a question that is their own but for case and spacing.
- **One caller's question shapes the answer the next caller is served.** The cache is shared by everyone who reaches the service, and the model answers the question as it was put. A question that carries an instruction of its own can leave an answer that follows it, and the second tier may then serve that answer to someone who asked the plain question. Nothing here tests that. It is one more reason for `exact` or `off` on a service that more than one person uses ([ADR 0015](0015-answers-over-http.md) on why the service is for one machine).
- Questions and their answers are now kept in the vector store, where before nothing but traces held them. They stay until the cache is emptied.
- The service loads a third model and holds it in memory. A machine short of memory pays for that: on the demo page, a minute after the checks above, the same hit by meaning took 1.2 seconds, 0.7 of them in the verifier.
- Changing any setting that decides an answer starts an empty cache. The old one stays in the store until `cache-clear --all`, and is served from again when the setting is put back.
- An evaluation is unchanged: it gets no cache, and its saved runs still describe the model.
