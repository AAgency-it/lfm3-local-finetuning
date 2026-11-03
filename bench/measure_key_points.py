"""Measure `key_points` on a test set (sentences already split) against a keyword baseline with no model (server on).

    python bench/measure_key_points.py <test.jsonl> [threshold]

Each line of the test file: {"id": ..., "frasi": [sentence, ...], "chiave": [indexes of the key sentences]}.
Scores per sentence are saved to results/, so the threshold can be re-evaluated without calling the model again.
The "per_threshold_optimistic" table is for diagnosis only: choosing a threshold on the test set is optimistic.
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from lfm3_local import client, extract as E, key_points as P  # noqa: E402

_WORDS = re.compile(r'\b(decis\w*|confermi\w*|approvat\w*|concordat\w*|entro|consegner\w*|invier\w*|preparer\w*|ci pens\w*|me ne occup\w*|'
                    r'bloccat\w*|blocca\w*|manca\w*|ritard\w*|slitt\w*|accettat\w*|fissat\w*|ci vediamo|prenot\w*)\b', re.I)


def keyword_baseline(sentence):
    return bool(_WORDS.search(sentence) or E.amounts_in_text(sentence) or E.dates_in_text(sentence))


def count(docs, decide):
    tp = fp = fn = 0
    for d in docs:
        true = set(d['chiave'])
        chosen = {i for i in range(len(d['frasi'])) if decide(d, i)}
        tp += len(true & chosen)
        fp += len(chosen - true)
        fn += len(true - chosen)
    pr = tp / (tp + fp) if tp + fp else 0.0
    re_ = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': round(pr, 3), 'recall': round(re_, 3), 'f1': round(2 * pr * re_ / (pr + re_), 3) if pr + re_ else 0.0, 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    threshold = float(args.pop()) if len(args) > 1 and re.fullmatch(r'0?\.\d+', args[-1]) else P.THRESHOLD
    if not client.healthy():
        sys.exit('server is off: start llama-server first (docs/install.md)')
    docs = [json.loads(r) for p in args for r in open(p, encoding='utf-8') if r.strip()]
    t0, scores = time.time(), {}
    for d in docs:
        scores[d['id']] = [P.score(d['frasi'], i) for i in range(len(d['frasi']))]
        if any(s is None for s in scores[d['id']]):
            sys.exit('server off or invalid answer: ' + d['id'])
    out = {'documents': len(docs), 'sentences': sum(len(v) for v in scores.values()), 'seconds': round(time.time() - t0, 1), 'threshold': threshold,
           'model': count(docs, lambda d, i: scores[d['id']][i] >= threshold),
           'keyword_baseline': count(docs, lambda d, i: keyword_baseline(d['frasi'][i])),
           'per_threshold_optimistic': {str(s): count(docs, lambda d, i, s=s: scores[d['id']][i] >= s) for s in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}}
    dest = HERE.parent / 'results'
    dest.mkdir(exist_ok=True)
    name = dest / f'key_points_{time.strftime("%Y%m%dT%H%M%S")}.json'
    name.write_text(json.dumps({'summary': out, 'scores': scores}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print('saved to', name)


if __name__ == '__main__':
    main()
