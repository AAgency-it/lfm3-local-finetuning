"""Build the dataset of the public generic adapter: synthetic data only, two tasks (date + amount extraction, log line filter).

Nothing here comes from our notes, our customers or a language model: every row is produced by data_synthetic/gen_extract.py and
data_synthetic/gen_logs.py from a public seed, and the label is born together with the text, so it is exact by construction.
  - training and dev: seeds from TRAIN_SEED (1 item in 9 goes to dev; log documents are split whole, never by line);
  - bench: seeds from BENCH_SEED, plus the 50 hand-written sentences of bench/extract_bench.jsonl;
  - a training or dev row whose prompt equals a bench prompt is dropped (the count is printed).
The extraction prompt is the one the skill uses (src/lfm3_local/extract.py). The log prompt is LOG_SYSTEM below: one line at a time,
with up to 3 lines before and 1 after, answer "rilevante" or "rumore".

Usage:
    python training/build_generic.py --out data_generic          # writes train.jsonl, dev.jsonl, bench.jsonl, baselines.json
then pack it for the cloud with training/pack_for_cloud.py (see docs/free-gpu.md).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data_synthetic'))
sys.path.insert(0, str(ROOT / 'src'))
import gen_extract  # noqa: E402
import gen_logs  # noqa: E402
from lfm3_local import extract  # noqa: E402

TRAIN_SEED, BENCH_SEED = 1_000_000, 2_000_000
LOG_SYSTEM = ("Classifica una riga di log. Rispondi solo «rilevante» o «rumore». È rilevante se è un errore, un fallimento, "
              "un'eccezione o il suo messaggio, il nome di un test che fallisce, la posizione nel file che ha causato il problema, "
              "un riepilogo finale (conteggi, esito, tempo totale, codice di uscita) o lo stato di un servizio (in ascolto, non parte). "
              "Tutto il resto è rumore: avanzamento, test passati, intestazioni, versioni, configurazione, righe di contesto del codice, "
              "avvisi di deprecazione, ripetizioni.")


def answer_json(data, amount):
    if amount is not None and float(amount).is_integer():
        amount = int(amount)
    return json.dumps({'data': data, 'importo_euro': amount}, ensure_ascii=False)


def _extract(text, data, amount, source):
    return {'task': 'extract', 'source': source, 'messages': [
        {'role': 'system', 'content': extract.SYSTEM},
        {'role': 'user', 'content': 'Frase: ' + text},
        {'role': 'assistant', 'content': answer_json(data, amount)}]}


def extract_row(seed):
    text, ans = gen_extract.make_sentence(seed)
    return _extract(text, ans['data'], ans['importo_euro'], f'gen_extract:{seed}')


def log_rows(seed):
    _, lines = gen_logs.make_document(seed)
    rows = []
    for i, (line, label) in enumerate(lines):
        before = '\n'.join(x for x, _ in lines[max(0, i - 3):i]) or '(inizio)'
        after = lines[i + 1][0] if i + 1 < len(lines) else '(fine)'
        rows.append({'task': 'logs', 'source': f'gen_logs:{seed}#{i}', 'messages': [
            {'role': 'system', 'content': LOG_SYSTEM},
            {'role': 'user', 'content': f'PRIMA:\n{before}\nRIGA: {line}\nDOPO:\n{after}'},
            {'role': 'assistant', 'content': label}]})
    return rows


def public_bench():
    out = []
    for x in open(ROOT / 'bench' / 'extract_bench.jsonl', encoding='utf-8'):
        if x.strip():
            r = json.loads(x)
            out.append(_extract(r['testo'], r['data'], r['importo_euro'], 'extract_bench:' + r['id']))
    return out


def build(n_extract=2000, n_log_docs=60, n_bench_extract=200, n_bench_log_docs=20):
    bench = [extract_row(BENCH_SEED + i) for i in range(n_bench_extract)] + public_bench()
    for i in range(n_bench_log_docs):
        bench += log_rows(BENCH_SEED + i)
    seen = {r['messages'][1]['content'] for r in bench}
    train, dev = [], []
    for i in range(n_extract):
        (dev if i % 9 == 8 else train).append(extract_row(TRAIN_SEED + i))
    for i in range(n_log_docs):
        (dev if i % 9 == 8 else train).extend(log_rows(TRAIN_SEED + i))
    dropped = sum(r['messages'][1]['content'] in seen for r in train + dev)
    train = [r for r in train if r['messages'][1]['content'] not in seen]
    dev = [r for r in dev if r['messages'][1]['content'] not in seen]
    return {'train': train, 'dev': dev, 'bench': bench, 'dropped_as_bench_duplicates': dropped}


def baselines(bench):
    """No-model baselines on the bench: regex alone for extraction (undecided = wrong), 'always noise' for logs."""
    ex = [r for r in bench if r['task'] == 'extract']
    logs = [r for r in bench if r['task'] == 'logs']
    right = 0
    for r in ex:
        got = extract.by_regex(r['messages'][1]['content'][len('Frase: '):])
        if got is not None and answer_json(got.get('data'), got.get('importo_euro')) == r['messages'][2]['content']:
            right += 1
    noise = sum(r['messages'][2]['content'] == gen_logs.N for r in logs)
    return {'extract_regex': round(right / len(ex), 3) if ex else 0.0, 'logs_majority': round(noise / len(logs), 3) if logs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data_generic')
    a = ap.parse_args()
    d = build()
    os.makedirs(a.out, exist_ok=True)
    for name in ('train', 'dev', 'bench'):
        with open(os.path.join(a.out, f'{name}.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in d[name])
        per = {t: sum(r['task'] == t for r in d[name]) for t in ('extract', 'logs')}
        print(name, len(d[name]), per)
    base = baselines(d['bench'])
    info = {'train_seed': TRAIN_SEED, 'bench_seed': BENCH_SEED, 'rows': {k: len(d[k]) for k in ('train', 'dev', 'bench')},
            'dropped_as_bench_duplicates': d['dropped_as_bench_duplicates'], 'baselines_on_bench': base}
    with open(os.path.join(a.out, 'baselines.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(info, f, indent=1)
    print(info)


if __name__ == '__main__':
    main()
