# Results (honest version)

Everything below was measured on one machine (a desktop with a GTX 1660 Ti, 6 GB), with llama.cpp b12002, thinking off,
temperature 0 unless stated. **The base model is Q4_K_M, the fine-tuned ones Q8_0**: part of every gap may come from quantisation,
not from training. One person judged the hand-scored items. These are indications, not a benchmark. Where a number comes from a file
in this repository, the file is named; the few that come from our lab notes are marked "lab notes" and are not published.

Names of customers, rule sets and task categories are replaced by labels (set A-D, cat 1-12) before a file is published
(`tools/publish_results.py`).

## Summary

| Job | Base LFM3 1.7B | After fine-tuning | No model | Verdict |
|---|---|---|---|---|
| Say whether a sentence breaks a rule set (4 sets x 80 sentences) | 0.537 | **0.956** | 0.50 (always one answer) | from a coin to a useful **alarm**, never a gate |
| Pick key sentences, synthetic test (60 docs, 1,680 sentences, F1) | 0.379 | **0.963** (round 1) | 0.734 (keywords) | big win, but the test comes from the training generator: a ceiling |
| Pick key sentences in real internal notes (58 docs, F1) | 0.088 | 0.444, precision 0.306 (round 2) | 0.223 (keywords) | beats keywords, drags along many lines: an index, not a summary |
| Extract date + amount (50 sentences we wrote, exact) | 0.68 | **0.96** (round 2) | 0.90 (regex) | regex first, model for the rest, validator always; 19 of the 50 share a training template: a ceiling |
| Label log lines relevant / noise (153 real lines) | 0.288 | **0.915** (round 2) | 0.725 (all noise) | useful as a filter, still misses some relevant lines |
| Guess a task's category from its title (170 tasks) | 0.194 | 0.729 (round 2) | 0.812 (label already written by hand) | **loses**: keep the hand label |

Sources: base `misura_base_prova_20261031T023952.json` (rules, categories), `misura_20261031T022041.json` (extraction),
`key_points_20261030T161116.json` (key points); round 1 `key_points_20261031T012510.json`; round 2 `misura_grande_*_20261031T*.json`
and `misura_grande_chiave_somma.json`; base on real notes and real logs, measured on 2026-11-03 with the same scripts and prompts as
round 2: `misura_base_f1_chiave_somma.json` and `misura_base_f1_20261103T165544.json`.

## Round 1: one skill (key sentences), synthetic data only

Rule fixed before the run: promote only above F1 0.80 on the test set, threshold 0.5 chosen in advance.

| | Precision | Recall | F1 |
|---|---|---|---|
| Base (Q4_K_M) | 0.59 | 0.28 | 0.379 |
| Keyword list | 0.696 | 0.776 | 0.734 |
| Fine-tuned (Q8_0) | 0.976 | 0.951 | **0.963** |

F1 barely moves between thresholds 0.2 and 0.8 (0.963-0.967): the model answers with near-certain scores.

## Round 2: several skills in one adapter

- **Rule sets.** Test and training sentences were both invented by a larger model from the same written rules; sentences equal or
  nearly equal to a test sentence were removed from training (`composizione.json`). Same family, so a ceiling. Per set: 0.963 / 0.988 / 0.925 / 0.95 (base: 0.512 / 0.625 / 0.512 / 0.5). The base let 30-40 of 40 forbidden sentences
  pass in every set; the fine-tuned model lets 0-1 pass. It is still a 1.7B model judging writing rules: in our stack it may only **add**
  warnings for a larger model, it never clears a text.
- **Extraction.** 1.00 on 200 generated sentences, 0.96 on the 50 we wrote (`bench/extract_bench.jsonl`). The regex alone: 0.88 and 0.90,
  always right where it decides. The base model on the same 200: 0.46. 19 of the 50 sentences follow one of the sentence templates in
  `data_synthetic/gen_extract.py`, which round 2 trained on (with other numbers): for the fine-tuned model this is a ceiling.
  The list: `python tools/check_template_overlap.py`; the count: `published_results/extract_template_overlap.json`.
- **Logs.** On 300 generated lines 0.993 ("all noise" 0.70): same generator as the training lines, so a ceiling. On 153 real lines 0.915: relevant lines 0.857, noise 0.937.
  The base model, with the same script and prompt (2026-11-03): 0.317 on the generated lines and 0.288 on the real ones; it calls almost
  everything relevant. An earlier measure with a different prompt (2026-10-31) gave the base 0.26 on the generated lines.
  About one relevant line in seven is still missed, so the big model always reads the end of the log as well.
- **Key sentences in real notes.** Two people marked the key sentences of 58 notes (agreement kappa 0.829 on 2,629 sentences); the bench uses
  the sentences both marked. F1 0.444 at threshold 0.5 (recall 0.807, precision 0.306), 0.466 at 0.7. The keyword list: 0.223. The base model: 0.088 (precision 0.054,
  recall 0.234); its 2 invalid answers, repeated identically on a second try with a healthy server, are counted as errors.
  None of the 58 notes, nor any file they come from, is among the documents used for training and dev.
  On the round 1 synthetic test the round 2 model keeps F1 0.967: it did not forget the first skill.
- **Categories.** 12 categories in all; the 170 scored tasks cover 9 of them (the other 3 appear only among the wrong answers), and the file
  records 19 more tasks left out because their right category is not known. 0.729 against 0.812 for the label people had already written. Some categories are easy (1.0), others are not (0.522).
  We do not use it for this.

## Base model only: what we measured before training (lab notes unless a file is named)

| Job | Result | Verdict |
|---|---|---|
| Extract date + amount, regex first, model only when ambiguous | 47 of 50 accepted, all 47 right (see below) | good, **with a validator** |
| Rewrite a short sentence in direct Italian | 7 of 10 good by hand; the automatic check accepted 9 of 10 | draft only: it changes meaning in about 1 case in 3 |
| One-line summary of a long text | 1 of 10 acceptable, 4 of 10 with a better prompt | do not use |

### Extraction: three runs, and what they teach

Files: `published_results/extract_*.json`. Bench: `bench/extract_bench.jsonl` (50 invented sentences).

| Run | Accepted | Right among accepted | Coverage | Precision | Median ms |
|---|---|---|---|---|---|
| Model only, validator on (`131232`) | 32 | 32 | 0.64 | 1.00 | 404 |
| Regex first, model for the ambiguous ones (`154928`) | 48 | 47 | 0.96 | 0.979 | 398 |
| Same, after fixing the failure seen in the previous run (`155002`) | 47 | 47 | 0.94 | 1.00 | 492 |

- The regex alone gets **45 of 50** right. The model adds a few hard cases; the main gain is that the validator makes wrong answers visible.
- The second run had one serious error: "the list price goes from 120 to 135 euro" returned 120. The fix was a rule for ranges, written after
  seeing the failure. So the third run is **not** a clean test: we tuned on the bench.
- **We wrote the bench ourselves, so it is optimistic.** Cases V1-01..V1-20 were seen while designing the prompt; V2-01..V2-30 were written afterwards without looking at the model's answers.
- The promotion rule we set before measuring: precision on accepted >= 0.95, coverage >= 0.40, no serious error.

### The model's confidence is not a filter
The first-token probability is almost always extreme: already at threshold 0.70, 93-100% of sentences pass (lab notes). Round 1 shows the
same after training: F1 is flat across thresholds.

## What we would tell a friend

1. Use it as a fast typist with a supervisor (a validator in code), not as a judge.
2. Put a regex in front. On our extraction bench it did most of the work for free.
3. Short, fixed-format inputs. Past about 500 characters the base model repeats the instruction or invents.
4. Measure against a no-model baseline. If you cannot beat the baseline, ship the baseline (we did, for categories).
5. Keep the test set out of everything: prompts, thresholds and training.
6. A synthetic test from your own generator is a ceiling. Build a small real bench, even 50 items, before you trust the number.

## Hardware and speed (our machine only)

Measured on 2026-11-03 on a GTX 1660 Ti (6,144 MiB), one server at a time, nothing else on the card: three separate starts of the base
server, a fixed prompt (40 English nouns, 256 tokens, temperature 0, the first of four requests discarded). File:
`published_results/hardware_f1_base_tre_avvii.json`.

| Base LFM3 1.7B, Q4_K_M | Measured |
|---|---|
| GPU memory taken by the server | 1,401 MiB (all three starts) |
| System RAM of the server process | 1,477-1,479 MB |
| Generation speed (median of each start) | 118.1, 100.6 and 123.4 tokens/s |
| Prompt processing | 416.9, 410.8 and 459.4 tokens/s |
| One 256-token answer | 2,122-2,608 ms |

The speed moves by about 20% from one start to the next on the same machine: read it as "about 100-120 tokens per second", not as a
precise figure. The two fine-tuned models (Q8_0) were **not** measured here: the run stopped on 2026-11-03, first because Windows Smart App
Control blocked one of llama.cpp's libraries for a moment, then because the machine ran low on memory. We give no estimate for them.
