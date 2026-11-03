"""Multi-task SFT LoRA of LFM3 1.7B: several small tasks in one adapter (our second run, 2026-10-31, on a free Kaggle T4).

Same recipe as train_sft.py (Liquid AI cookbook: LoRA r=16, alpha=32, dropout 0.05, lr 2e-4, micro-batch 4 x accumulation 4,
thinking OFF also in training, adapters in fp32 with an fp16 GradScaler), with these differences:
 - data: every train*.jsonl in this script's folder ("task" and "messages" fields) plus dev.jsonl; the loss counts only on the answer;
 - length-bucketed batches (some prompts are 3x longer than others): less padding, same recipe;
 - per-task evaluation after each epoch with "forced exact match": every token of the answer (and the end of text) must be the
   model's most likely token given the prompt and the correct previous tokens. It equals greedy generation reproducing the exact
   answer, without calling generate();
 - each epoch's adapter is saved (out/adapters/epN) and kept in memory; at the end the epoch with the best task average is merged.
Our run: 13,486 training rows, 3 epochs, 12,160.7 s on one T4; dev average 0.385 before, 0.948 after.

Install (Kaggle):  pip uninstall -y -q torchao; pip install -q -U transformers peft accelerate
Run:               python train_multi.py [--epochs 3] [--out lfm3-multi] [--limit 0]
Never real customer data in here. Our run mixed synthetic data with internal notes; the public adapter is trained on synthetic data only.
"""
import argparse
import glob
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = 'LiquidAI/LFM3-1.7B'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj', 'out_proj', 'w1', 'w2', 'w3']
HERE = os.path.dirname(os.path.abspath(__file__))


def read_files(paths):
    out = []
    for p in paths:
        out += [json.loads(r) for r in open(p, encoding='utf-8') if r.strip()]
    return out


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


def length_buckets(data, micro, rnd, window=256):
    """Shuffle, sort by length inside windows of `window` examples, cut into micro-batches, shuffle the micro-batches."""
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    groups = []
    for k in range(0, len(idx), window):
        part = sorted(idx[k:k + window], key=lambda i: len(data[i][0]))
        groups += [part[j:j + micro] for j in range(0, len(part), micro) if len(part[j:j + micro]) == micro]
    rnd.shuffle(groups)
    return groups


@torch.no_grad()
def evaluate(model, tok, dev, device, size=8):
    """Forced exact match per task; returns ({task: share}, average over tasks)."""
    model.eval()
    per = {}
    rows = sorted(dev, key=lambda e: len(e['_ids']))
    for k in range(0, len(rows), size):
        group = rows[k:k + size]
        ids, att, lab = collate(tok, [(e['_ids'], e['_lab']) for e in group], device)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(input_ids=ids, attention_mask=att).logits
        pred = logits[:, :-1].argmax(-1)
        target = lab[:, 1:]
        for j, e in enumerate(group):
            m = target[j] != -100
            ok = bool(m.any()) and bool((pred[j][m] == target[j][m]).all())
            s = per.setdefault(e['task'], [0, 0])
            s[0] += ok
            s[1] += 1
    model.train()
    acc = {t: round(a / n, 4) for t, (a, n) in per.items()}
    return acc, round(sum(acc.values()) / len(acc), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--out', default='lfm3-multi')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0, help='quick test: use only the first N train and dev examples')
    ap.add_argument('--no_merge', action='store_true', help='keep only the LoRA adapters (out/adapters/epN), skip the merged model')
    a = ap.parse_args()
    rnd = random.Random(7)
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
    train = read_files(sorted(glob.glob(os.path.join(HERE, 'train*.jsonl'))))
    dev = read_files([os.path.join(HERE, 'dev.jsonl')])
    if a.limit:
        train, dev = train[:a.limit], dev[:a.limit]
    data = [encode(tok, e['messages'], a.max_len) for e in train]
    for e in dev:
        e['_ids'], e['_lab'] = encode(tok, e['messages'], a.max_len)
    print('training examples:', len(data), 'dev:', len(dev))
    bench_path = os.path.join(HERE, 'bench.jsonl')   # optional held-out bench: measured on the base before training and on the adapter after
    bench = read_files([bench_path]) if os.path.exists(bench_path) else []
    for e in bench:
        e['_ids'], e['_lab'] = encode(tok, e['messages'], a.max_len)
    metrics = {}
    acc, avg = evaluate(model, tok, dev, device)
    metrics['dev_before'] = {'per_task': acc, 'average': avg}
    print('dev before:', metrics['dev_before'], flush=True)
    if bench:
        acc, avg = evaluate(model, tok, bench, device)
        metrics['bench_base'] = {'per_task': acc, 'average': avg, 'rows': len(bench)}
        print('bench, base model:', metrics['bench_base'], flush=True)
    micro, accum = 4, 4
    steps = (len(data) // (micro * accum)) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(steps * 0.05), steps)
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    step, t0 = 0, time.time()
    best, best_ep, copies = -1.0, 0, {}
    os.makedirs(os.path.join(a.out, 'adapters'), exist_ok=True)
    for ep in range(a.epochs):
        groups = length_buckets(data, micro, rnd)
        for k in range(0, len(groups) - accum + 1, accum):
            total = 0.0
            for g in groups[k:k + accum]:
                ids, att, lab = collate(tok, [data[i] for i in g], device)
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
            if step % 20 == 0:
                print(f'epoch {ep + 1} step {step}/{steps} loss {total:.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
        acc, avg = evaluate(model, tok, dev, device)
        metrics[f'dev_epoch_{ep + 1}'] = {'per_task': acc, 'average': avg}
        print(f'dev after epoch {ep + 1}:', metrics[f'dev_epoch_{ep + 1}'], flush=True)
        model.save_pretrained(os.path.join(a.out, 'adapters', f'ep{ep + 1}'))
        copies[ep + 1] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}
        if avg > best:
            best, best_ep = avg, ep + 1
        json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    if best_ep != a.epochs:
        print(f'merging epoch {best_ep} (average {best}) instead of the last one', flush=True)
        model.load_state_dict(copies[best_ep], strict=False)
    metrics['merged_epoch'] = best_ep
    metrics['training_seconds'] = round(time.time() - t0, 1)
    if bench:
        acc, avg = evaluate(model, tok, bench, device)
        metrics[f'bench_adapter_ep{best_ep}'] = {'per_task': acc, 'average': avg, 'rows': len(bench)}
        print('bench, adapter:', metrics[f'bench_adapter_ep{best_ep}'], flush=True)
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    if not a.no_merge:
        merged = model.merge_and_unload().to(torch.float16)
        merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    print('done:', metrics)


if __name__ == '__main__':
    main()
