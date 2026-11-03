# Pitfalls we actually hit

Each one cost us time. Where a number comes from a published file, the file is named; numbers marked *(lab notes)* come from our
working notes, are not published and were measured in other conditions: do not compare them with `results.md`.

## 1. The answer comes back empty: thinking is on
LFM3 is a reasoning model. Without `chat_template_kwargs: {"enable_thinking": false}` the reasoning ends up in `reasoning_content`
and `content` is empty (or your token budget is spent before the answer). Set it on the server (`--chat-template-kwargs '{"enable_thinking": false}'`)
and in every request (`src/lfm3_local/client.py` does). Do the same **in training**: the chat template must be applied with `enable_thinking=False`
or the model learns a format it will not see at inference (`training/train_sft.py`).

## 2. The llama.cpp build matters
An older build we already had (b11081) did not know the `lfm3` architecture. We used **b12002** (CUDA 12.4 build). If the server dies at load
with an unknown-architecture error, update llama.cpp before debugging anything else.

## 3. Regex first, model second
For extraction the regexes alone got 45 of 50 on our bench. Starting from the model and "fixing" it with code is the wrong order:
start from the code and give the model only what the code cannot decide (`src/lfm3_local/extract.py`).

## 4. The model proposes, code disposes
Every skill here has a deterministic check. The extraction validator accepts a date or an amount only if it is written in the sentence.
Without it, the model invented a date in one hand test. A check does not need to be smart; it needs to be independent of the model.

## 5. The test set never goes into training (or prompt tuning)
Split by document, not by sentence, or neighbouring sentences leak. `training/build_dataset.py` also stops if more than 2 sentences
in 100 of the training documents appear in the test set. We tuned a rule on our extraction bench after seeing a failure: the result is
reported as "not a clean test" (see `results.md`).

## 6. Confidence from log-probabilities is not a safety filter
With a constrained answer the first token's probability is almost always extreme, so thresholds filter nothing useful
(93-100% of sentences passed even at 0.70, *lab notes*). It is a weak diagnostic at best.

## 7. Grammar-constrained labels with more than one token
Constraining the output with a grammar is great for format, but a label made of several tokens can bias the choice towards one label.
We saw a prompt that sorts short titles into categories collapse onto one label (0.206 accuracy) and recover with a second prompt
(0.700, *lab notes*, a different bench from the one in `results.md`). We did not isolate how
much was the model and how much was the shape of the constraint. Test your labels one by one.

## 8. A shared server inflates your timings
We measured while the server was also serving another job: one request out of 980 took 5 s (*lab notes*). Measure latency on an idle server, and report the median.

## 9. Long texts
Past roughly 500 characters, one-line summaries failed (1 of 10 acceptable, *lab notes*). Split the problem: one short yes/no question per sentence
(`src/lfm3_local/key_points.py`) works far better than "summarise this".

## 10. Memory
A 6 GB card holds this model (1,401 MiB of GPU memory at Q4_K_M, measured on 2026-11-03: see `results.md`) with room to spare, but not a second, larger local model next to it. Training is another matter:
we trained in the cloud (see `finetuning.md`), not on the desktop card.

## 11. The server dies silently inside a memory-capped job
We run heavy jobs inside a Windows job object with a memory limit. Started inside it, llama-server exited during CUDA start-up without writing
a single log line; started outside, the same command was healthy. Start the server outside any memory cap and cap only the client that
sends the requests. Then check `/health` before measuring, and redo any measurement that lost an answer.

## 12. `--cache-ram 0`
*(Lab notes.)* Inside a memory-capped job a server ran for 15 minutes and then died without an error line, and the measurement counted the lost answers as
model mistakes. Probable cause, not fully proven: the prompt cache that recent llama.cpp builds keep in system RAM (that process had grown to
4.3 GB of committed memory). Restarted with `--cache-ram 0` it stayed at about 2.5 GB. We now start every server with `--cache-ram 0`,
check `/health` before reading any number, and count lost answers separately from wrong ones.
