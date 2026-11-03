# Fine-tuning LFM3 1.7B on a free GPU: our two runs

Both runs: LoRA on one free Kaggle T4, 0 euro, recipe from the Liquid AI cookbook (r=16, alpha=32, dropout 0.05, lr 2e-4,
micro-batch 4 x accumulation 4, fp16 with fp32 adapters, thinking off also in training, loss only on the answer).
Run facts: `published_results/training_runs.json`. How to get the GPU: [free-gpu.md](free-gpu.md).

## Round 1 (2026-10-30): one skill, synthetic data only

- **Question, fixed before the run:** can a free T4 turn a base model that loses to a keyword list into one that beats it?
  Promotion rule: F1 above **0.80** on the test set (60 documents, 1,680 sentences, never seen in training).
- **Data:** about 90 invented documents (minutes, emails, reports) with their key sentences marked by the writer, written by a larger model
  with the instruction to use no real names, firms or figures. One example per sentence (`PRIMA / TARGET / DOPO` prompt, answer `chiave`
  or `altro`): 2,485 training and 338 dev examples, split by document. `training/build_dataset.py` builds them and stops on overlaps.
- **Training:** `training/train_sft.py`, 2 epochs, 310 optimizer steps, about 20 minutes.
- **Result** (`published_results/key_points_20261031T012510.json`, threshold 0.5 chosen in advance):

| | Precision | Recall | F1 |
|---|---|---|---|
| Base model (Q4_K_M) | 0.59 | 0.28 | 0.379 |
| Keyword list, no model | 0.696 | 0.776 | 0.734 |
| Fine-tuned (Q8_0) | 0.976 | 0.951 | **0.963** |

- **Limit:** test and training documents come from the same family of generator, so 0.963 is a ceiling. On real internal notes it is much lower (round 2 below).

## Round 2 (2026-10-31): several small skills in one adapter

- **Data:** 13,486 training and 1,277 dev rows in five task families (counts in `published_results/composizione.json`). What went to the
  cloud, task by task:
  - *key sentences:* the synthetic documents of round 1, 63 new synthetic documents in the style of internal notes, and 113 documents
    from our internal Obsidian knowledge base (process notes and work logs; 47 more were left out, see below);
  - *log triage* and *extraction:* the generators in `data_synthetic/`, nothing else;
  - *rule checks:* short texts invented from the written rules of four brands we work for (no published posts, no personal data);
  - *task categories:* task titles invented using the codes and the vocabulary of our customers' fields of work;
    2 that were equal or nearly equal to a real title were removed.
- **Two clouds, not one.**
  - *Kaggle* (training) received the training and dev files above.
  - *Claude* (Anthropic's hosted model, the same one we use as orchestrator, through Claude Code) wrote the invented rows: it received
    the written rules of the four brands, the category codes and vocabulary, and the 113 + 14 internal documents, whose key sentences it
    marked. It reads those notes in our daily work anyway.
- **How the internal documents were filtered.** Two pattern filters, no human reading: whole files about customers, prospects or our
  sales software were left out (by name and folder, 47 documents); inside the rest, every sentence with an e-mail, a phone number, a
  legal identifier, an address, a secret or a person's name (outside a short allowed list: our team and a few partners) was dropped,
  and the final package was scanned again line by line. We did not re-read the 113 documents one by one, so a customer named in a work
  log in a form the patterns miss cannot be ruled out. The invented rows speak about our customers' kind of work: the data and the
  weights stay private.
- **Training:** `training/train_multi.py`, 3 epochs, 2,526 optimizer steps, about 3 h 20 min (transcribed by hand from the notebook log). Length-bucketed batches, per-task
  "forced exact match" on dev after each epoch, best epoch merged (it was the last). Dev average 0.3853 before, 0.9479 after:
  the dev set comes from the same sources as training (mostly synthetic, plus 14 documents of real notes, `composizione.json`), so again a ceiling.
- **Results on the benches** (base Q4_K_M vs fine-tuned Q8_0 vs no model): see [results.md](results.md). It wins on rule checks (a ceiling:
  same family as training); on real logs and real notes it beats both the base model and the no-model baseline; on extraction the test
  shares templates with the training generator (a ceiling); it loses on task categories against the label people had already written.

## What you need to repeat it
A Hugging Face login (to download the base weights), a Kaggle account with GPU access, your own synthetic data, and
`training/pack_for_cloud.py` to split the training file under the upload limit. Convert the merged model to GGUF with llama.cpp's tools,
quantise it, serve it, then measure it with the same bench and the same no-model baseline as the base model.
Do **not** train on customer or personal data in a free cloud notebook.

## Known limits
- The base is Q4_K_M and the fine-tuned models are Q8_0: part of every gap may come from quantisation.
- transformers/peft APIs move. Our round 2 install line on Kaggle was `pip uninstall -y -q torchao; pip install -q -U transformers peft accelerate`:
  it installs the latest versions, and **we did not record which ones** (torch, transformers, peft, accelerate). Print and pin them in your run
  (see [free-gpu.md](free-gpu.md), step 4).
- We publish neither weights nor datasets of these two runs. A generic adapter trained only on new synthetic data is planned and will be
  published only if it beats the base model and the license allows it.
