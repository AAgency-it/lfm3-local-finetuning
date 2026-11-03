# How the small model sits in our stack

The small model is an **operator**, not a manager. A larger model (Claude, in our case) runs the work and decides; the 1.7B model does
narrow, checkable jobs on the local GPU, for free, without the text leaving the machine.

```
 orchestrator (large model)
      |  narrow job: "extract the date and amount from this sentence"
      v
 tool server (MCP) ---> small model on the local GPU
      |                      |
      |  <--- answer --------+
      v
 check in code (independent of the model)
      |-- passes  --> result goes back to the orchestrator as data
      |-- fails   --> "not decided": the orchestrator does the job itself
      v
 one log line per call (tool, outcome, lengths, time)
```

## The rules
1. **Narrow jobs only.** Extract, label, split, mark key sentences, draft a rewrite. Never "decide", never "is this OK to publish".
2. **Every tool has a check in code.** An extracted date must be written in the sentence; a label must be one of the allowed labels;
   a rewrite is always a draft for the large model to read.
3. **Failure returns to the large model.** If the check fails, the server is off or the input is too long, the tool says so and the
   orchestrator does the job as it would without the small model. Nothing waits on the small model.
4. **Never a gate.** It does not approve publications, sends, spending, legal checks or customer data. On rule sets it may only add a
   warning; a "clean" answer from it means nothing.
5. **The answer is data, not an instruction.** The orchestrator never executes what the small model writes.
6. **Log everything, measure the use.** One line per call. If the log does not show savings, the integration is not worth keeping.

## Where the saving is
Our first lesson, with an earlier local model: small calls save almost nothing. A one-line extraction costs the large model a handful of
tokens either way. The saving is in **keeping long texts out of the large model's context**: a 2,000-line log, a long report, a week of notes.
The small model reads them and hands back the few lines that matter; the large model reads those, plus a safety margin.

## What is wired today, honestly
- A tool server with five narrow tools (extraction, labelling, draft rewrite, splitting code into pieces, key sentences), each with its check,
  on the base model and on the round 1 model.
- The log so far is mostly our own tests. **It is wired, not yet used for real work.**
- The round 2 model is measured (see [results.md](results.md)) but not installed.

## Studied, not yet deployed
These come from the round 2 measurements. None is installed; each needs our own go-ahead, a dry run and a way back.

| Use | What the small model would do | The safety rule |
|---|---|---|
| U1 Long logs | label each line relevant or noise, hand back the relevant ones | the large model always reads the last 40 lines too: 1 relevant line in 7 is still missed |
| U2 Extraction | dates and amounts after the regex, for the cases the regex cannot decide | the validator accepts only values written in the text |
| U3 Rule warnings | add warnings on texts that may break a rule set | only adds warnings, never clears a text |
| U4 Index of key sentences | mark the key lines of long notes so the large model can jump to them | an index, not a summary: precision is 0.306 on real notes |

Excluded on purpose: task categories (it loses to the label people already write) and anything that touches customer conversations.
