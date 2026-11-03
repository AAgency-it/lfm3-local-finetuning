# How to evaluate a small model for your own job

This is the method we used. It is the part of the repository that transfers to any model.

## 1. Decide the promotion rule before you measure
Write down the numbers that mean "use it" and the ones that mean "do not", before you look at a single result.
Ours for a skill with a validator: **precision on accepted answers >= 0.95, coverage >= 0.40, no serious error.**
For a model trained to beat a baseline: **F1 above the best no-model baseline** on the test set (ours: above 0.80 against a keyword baseline at 0.734).
A model that does not pass is not used, however much work went into it.

## 2. Always measure a baseline with no model
Regexes, keyword lists, a lookup. If your model does not beat it, ship the baseline. Our extraction regexes alone got 45 of 50; our
keyword baseline for key sentences got F1 0.734 and the base model got 0.379.

## 3. Three sets, three jobs
- **Development set**: where you tune prompts and thresholds.
- **Test set**: you look at it once per candidate. It never enters training, prompts or threshold selection.
- **Training set** (if you fine-tune): split by *document*, not by sentence.
Our extraction bench broke this rule in part (we fixed a rule after seeing a failure), and we say so in `results.md`.

## 4. Make wrong answers visible: a validator
A deterministic check that accepts only what can be verified (the date is in the sentence; the numbers are the same; the code compiles).
The measure that matters is **precision among accepted** and **coverage** (share accepted), not raw accuracy.
`bench/measure_extract.py` prints both and the baseline.

## 5. Report the awkward numbers
Per-case results, serious errors listed by id, which items you looked at while designing, who wrote the bench and who judged the hand-scored items.
One judge and a bench written by the authors are limits; say so next to the number.

## 6. Measure speed on an idle server
Report the median, not the mean, and note if the server was shared.

## 7. Check the model's confidence before trusting it
Compute precision at several confidence thresholds. If almost everything passes at 0.70, the confidence carries no information
(it did not, for us).

## Bench file format
`bench/extract_bench.jsonl`: one JSON per line, `{"id", "testo", "data", "importo_euro", "difficile"}`. Sentences are invented, in Italian.
`bench/measure_key_points.py` expects `{"id", "frasi": [...], "chiave": [indexes]}`.
