"""SFT LoRA of LFM3 1.7B on `key_points` (run it on a cloud GPU such as a free Kaggle/Colab T4, not on a 6 GB desktop card).

This is the script of our first run (2026-10-30, free Kaggle T4, 2 epochs, 310 optimizer steps, about 20 minutes, 0 euro):
2,485 training and 338 dev examples from 90 synthetic documents, split by document. Pin your library versions:
transformers/peft APIs move. Our second run (train_multi.py) used this install line on Kaggle:
`pip uninstall -y -q torchao; pip install -q -U transformers peft accelerate`.

Recipe from Liquid AI's cookbook (lfm3_sft_with_trl.ipynb): LoRA r=16, alpha=32, dropout 0.05, lr 2e-4, batch 4 x accumulation 4,
thinking OFF also in training, then merge. The loop is written with transformers + peft (no TRL, whose API changes between
versions) and the loss counts only on the answer ("chiave" / "altro"). Data: train*.jsonl and dev.jsonl ("messages" field),
in the same folder as this script. Changed after our run (2026-11-03): it now reads every train*.jsonl, so it accepts the split
files written by pack_for_cloud.py (train1.jsonl, train2.jsonl, ...); our run used a single train.jsonl.

Install (Colab/Kaggle):  pip install -U transformers peft accelerate datasets
Run:                     python train_sft.py [--epochs 2] [--out lfm3-key-points]
Output: folder `out` with the merged fp16 model (config, safetensors weights, tokenizer) + metrics.json (dev accuracy before/after).
Never real customer data in here: only the synthetic dataset built by build_dataset.py.
"""
import argparse
import glob
import json
import os
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = 'LiquidAI/LFM3-1.7B'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj', 'out_proj', 'w1', 'w2', 'w3']
HERE = os.path.dirname(os.path.abspath(__file__))


def read(pattern):
    """Every row of every file matching pattern in this folder, files in name order."""
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, pattern))):
        rows += [json.loads(r) for r in open(f, encoding='utf-8') if r.strip()]
    return rows


def encode(tok, messages, max_len):
    """(ids, labels): the prompt is masked (-100), the answer + end of text count."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    answer = messages[-1]['content'] + tok.eos_token
    p = tok(prompt, add_special_tokens=False)['input_ids']
    a = tok(answer, add_special_tokens=False)['input_ids']
    ids = (p + a)[:max_len]
    return ids, ([-100] * len(p) + a)[:max_len]


def collate(tok, examples, device):
    m = max(len(i) for i, _ in examples)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([i + [pad] * (m - len(i)) for i, _ in examples], device=device)
    lab = torch.tensor([l + [-100] * (m - len(l)) for _, l in examples], device=device)
    att = torch.tensor([[1] * len(i) + [0] * (m - len(i)) for i, _ in examples], device=device)
    return ids, att, lab


@torch.no_grad()
def accuracy(model, tok, dev, device):
    """Share of examples where, after the prompt, the more likely token between "chiave" and "altro" is the right one."""
    model.eval()
    id_key = tok('chiave', add_special_tokens=False)['input_ids'][0]
    id_other = tok('altro', add_special_tokens=False)['input_ids'][0]
    right = 0
    for e in dev:
        prompt = tok.apply_chat_template(e['messages'][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
        logit = model(input_ids=ids).logits[0, -1]
        right += ('chiave' if logit[id_key] > logit[id_other] else 'altro') == e['messages'][-1]['content']
    model.train()
    return round(right / len(dev), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--out', default='lfm3-key-points')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_len', type=int, default=1024)
    a = ap.parse_args()
    random.seed(7)
    torch.manual_seed(7)
    device = 'cuda'
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type='CAUSAL_LM'))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # adapters in fp32: needed by the fp16 grad scaler
    model.print_trainable_parameters()
    train, dev = read('train*.jsonl'), read('dev.jsonl')
    data = [encode(tok, e['messages'], a.max_len) for e in train]
    metrics = {'dev_before': accuracy(model, tok, dev, device)}
    print('dev accuracy before:', metrics['dev_before'])
    micro, accum = 4, 4
    steps = (len(data) // (micro * accum)) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.05), steps)
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    step = 0
    for ep in range(a.epochs):
        random.shuffle(data)
        for k in range(0, len(data) - micro * accum + 1, micro * accum):
            total = 0.0
            for j in range(accum):
                ids, att, lab = collate(tok, data[k + j * micro:k + (j + 1) * micro], device)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
                scaler.scale(loss).backward()
                total += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 10 == 0:
                print(f'epoch {ep + 1} step {step}/{steps} loss {total:.4f}')
        metrics[f'dev_after_epoch_{ep + 1}'] = accuracy(model, tok, dev, device)
        print(f'dev accuracy after epoch {ep + 1}:', metrics[f'dev_after_epoch_{ep + 1}'])
    merged = model.merge_and_unload().to(torch.float16)
    os.makedirs(a.out, exist_ok=True)
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print('done:', metrics)


if __name__ == '__main__':
    main()
