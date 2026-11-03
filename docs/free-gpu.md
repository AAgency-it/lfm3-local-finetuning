# Training on a free GPU (Kaggle)

Both our runs used one free NVIDIA T4 in a Kaggle notebook, at 0 euro. This page is the route we followed, not an official guide:
Kaggle changes its menus and quotas, so check their docs ([kaggle.com/docs](https://www.kaggle.com/docs)) when something does not match.

## Quota
Kaggle gives a weekly GPU quota and a time limit per session. **We do not quote a number here**: we could not confirm the current figure
on Kaggle's own documentation, and it has changed over the years. Your remaining quota is shown in your Kaggle account. For scale:
round 1 took about 20 minutes, round 2 about 3 h 20 min (`published_results/training_runs.json`).

## Steps
1. **Data.** Build `train.jsonl` and `dev.jsonl` on your machine (synthetic, or data you are allowed to send to a cloud).
   `python training/pack_for_cloud.py <folder> <upload folder>` splits them under the upload limit and copies the training script next to them.
2. **Dataset.** Upload the folder as a **private** Kaggle dataset.
3. **Notebook.** New notebook, *Add Input* → your dataset, *Settings* → *Accelerator* → GPU T4, *Internet* on (for `pip` and for the base
   weights). Our scripts downloaded `LiquidAI/LFM3-1.7B` without a login. If Hugging Face asks you for one, add the token as a Kaggle
   secret named `HF_TOKEN` (*Add-ons* → *Secrets*), never in a cell, and load it in the first cell (we did not need this; untested):
   ```
   import os
   from kaggle_secrets import UserSecretsClient
   os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
   ```
4. **Install, and write down the versions.** Our round 2 line, then the versions (we did not record ours, a limit of our runs):
   ```
   !pip uninstall -y -q torchao; pip install -q -U transformers peft accelerate
   !python -c "import torch, transformers, peft, accelerate; print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)"
   ```
5. **Run.** The dataset is mounted read-only under `/kaggle/input/<your-dataset>/`; the script reads the `train*.jsonl` and `dev.jsonl`
   next to it and writes where `--out` says, so point that to `/kaggle/working/`:
   ```
   !python /kaggle/input/<your-dataset>/train_multi.py --out /kaggle/working/lfm3-multi
   ```
   Then *Save Version* → *Save & Run All*, so the training does not depend on your browser staying open. Our notebook cells were not
   saved word by word: these lines follow the scripts and Kaggle's standard layout.
6. **Bring it home.** When the version has finished, open it, go to *Output* and download the `lfm3-multi` folder (we used the Download
   button). Then convert it to GGUF, quantise, serve it ([convert-and-serve.md](convert-and-serve.md)), and measure it on the same bench
   as the base model, with the same no-model baseline.

## What not to send
Anything you would not post publicly: customer data, personal data, credentials, internal documents. A free notebook is someone else's
computer. Our round 2 used documents from our own internal notes after a filter that left out every file about customers and prospects; what
went to the cloud, task by task, is in [finetuning.md](finetuning.md). The published adapter (planned) will use synthetic data only.
