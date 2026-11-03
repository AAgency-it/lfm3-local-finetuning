"""Measure the `extract` skill against bench/extract_bench.jsonl and against a regex-only baseline (needs the server on).

    python bench/measure_extract.py            -> results/extract_<timestamp>.json

Reports, per case: accepted by the validator / escalated, and among the accepted how many are right.
The baseline without a model is always printed: if the model does not beat it, you do not need the model.
Promotion rule we used: precision on accepted >= 0.95, coverage >= 0.40, no serious error.

HONESTY NOTE: bench cases V1-01..V1-20 were seen while the prompt was designed; V2-01..V2-30 were written afterwards without looking
at the model's answers. The bench was written by us, so it is optimistic. Do not tune prompts on it; keep a separate dev set.
"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from lfm3_local import client, extract as E  # noqa: E402


def read(path):
    return [json.loads(r) for r in open(path, encoding='utf-8') if r.strip()]


def baseline(text):
    """Regex only: the single date and the single amount, if there is exactly one."""
    d, a = E.dates_in_text(text), E.amounts_in_text(text)
    return {'data': d[0] if len(d) == 1 else None, 'importo_euro': a[0] if len(a) == 1 else None}


def right(r, expected):
    ok_d = r.get('data') == expected['data']
    a, b = r.get('importo_euro'), expected['importo_euro']
    return ok_d and ((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 0.005))


def main():
    if not client.healthy():
        sys.exit('server is off: start llama-server first (docs/install.md)')
    rows, base_right = [], 0
    for c in read(HERE / 'extract_bench.jsonl'):
        r = E.run(c['testo'])
        expected = {'data': c['data'], 'importo_euro': c['importo_euro']}
        accepted = r['outcome'] == 'accepted'
        rows.append({'id': c['id'], 'accepted': accepted, 'right': accepted and right(r, expected), 'ms': r.get('ms'),
                     'source': r.get('source'), 'hard': c.get('difficile', False)})
        base_right += right(baseline(c['testo']), expected)
    acc = [x for x in rows if x['accepted']]
    out = {'cases': len(rows), 'accepted': len(acc), 'right_among_accepted': sum(x['right'] for x in acc),
           'coverage': round(len(acc) / len(rows), 3), 'precision': round(sum(x['right'] for x in acc) / len(acc), 3) if acc else None,
           'baseline_regex_right': base_right, 'baseline_regex_of': len(rows),
           'median_ms': round(statistics.median([x['ms'] for x in rows if x['ms']])) if any(x['ms'] for x in rows) else None,
           'serious_errors': [x['id'] for x in acc if not x['right']], 'detail': rows, 'when': f'{datetime.now():%Y-%m-%dT%H:%M:%S}'}
    dest = HERE.parent / 'results'
    dest.mkdir(exist_ok=True)
    (dest / f'extract_{time.strftime("%Y%m%dT%H%M%S")}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'detail'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
