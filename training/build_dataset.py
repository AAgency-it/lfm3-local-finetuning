"""Build the SFT dataset for `key_points` from synthetic training documents.

One example = one sentence: the same prompt as src/lfm3_local/key_points.py (PRIMA / TARGET / DOPO), answer "chiave" or "altro".
Split by DOCUMENT, never by sentence: 1 document in 9 goes to the development set, the rest to training.
The test set never enters. Sentences identical to one in the test set are not used as targets; if the overlap is above 2 sentences
per 100 the script stops.

Input documents (you generate them, see docs/finetuning.md): JSONL, one per line
    {"id": "...", "frasi": [sentence, ...], "chiave": [indexes of the key sentences]}
Use only SYNTHETIC text. Never put real customer or personal data in a training set that goes to a cloud GPU.

Usage:
    python training/build_dataset.py --train docs_train/*.jsonl --test bench_test/*.jsonl --out data/
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lfm3_local import key_points as P  # noqa: E402


def read(paths):
    docs = []
    for p in paths:
        docs += [json.loads(r) for r in open(p, encoding='utf-8') if r.strip()]
    return docs


def examples(doc, forbidden):
    key = set(doc['chiave'])
    for i in range(len(doc['frasi'])):
        if doc['frasi'][i].strip().lower() in forbidden:
            continue
        yield {'id': f"{doc['id']}#{i}", 'messages': [
            {'role': 'system', 'content': P.SYSTEM},
            {'role': 'user', 'content': P.sentence_prompt(doc['frasi'], i)},
            {'role': 'assistant', 'content': 'chiave' if i in key else 'altro'}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    train_docs, test = read(a.train), read(a.test)
    test_sentences = {f.strip().lower() for d in test for f in d['frasi']}
    total = sum(len(d['frasi']) for d in train_docs)
    shared = sum(f.strip().lower() in test_sentences for d in train_docs for f in d['frasi'])
    if shared > total * 0.02:
        sys.exit(f'STOP: {shared} of {total} test-set sentences are inside the training documents')
    print(f'training sentences equal to a test sentence (dropped as targets): {shared} of {total}')
    train, dev = [], []
    for n, d in enumerate(sorted(train_docs, key=lambda x: x['id'])):
        (dev if n % 9 == 8 else train).extend(examples(d, test_sentences))
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (('train', train), ('dev', dev)):
        with open(f'{a.out}/{name}.jsonl', 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        pos = sum(r['messages'][-1]['content'] == 'chiave' for r in rows)
        print(f'{name}: {len(rows)} examples, {pos} key ({pos / len(rows):.1%})')
    print('training documents:', len(train_docs))


if __name__ == '__main__':
    main()
