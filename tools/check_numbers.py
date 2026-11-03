"""Every headline number in the READMEs must come from a published result file.

    python tools/check_numbers.py            (reads published_results/claims.json)

claims.json is a list of {"key", "file", "path", "value", "digits"}: the number `value`, rounded to `digits`, must equal the
value found at `path` (dot-separated) inside published_results/<file>, and must appear in README.md (dot decimal) and in
README.it.md (comma decimal). The figures are drawn from the same claims.json, so README, figures and results cannot drift.
Exit code 1 if anything disagrees.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lookup(obj, path):
    """`path` is "a.b.c", or a list ["a", "0.5", "c"] when a key itself contains a dot."""
    for part in (path if isinstance(path, list) else path.split('.')):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def shown(value, digits, comma):
    s = f'{value:,.{digits}f}' if digits else f'{value:,.0f}'
    if comma:
        s = s.replace(',', '\0').replace('.', ',').replace('\0', '.')
    return s


def verify(claims, root=ROOT):
    problems = []
    readmes = {name: (root / name).read_text(encoding='utf-8') for name in ('README.md', 'README.it.md')}
    for c in claims:
        try:
            real = lookup(json.loads((root / 'published_results' / c['file']).read_text(encoding='utf-8')), c['path'])
        except (OSError, KeyError, IndexError, ValueError) as e:
            problems.append(f"{c['key']}: cannot read {c['file']}:{c['path']} ({e})")
            continue
        if round(float(real), c['digits']) != round(float(c['value']), c['digits']):
            problems.append(f"{c['key']}: value {c['value']} but {c['file']}:{c['path']} says {real}")
            continue
        if c.get('readme', True):
            for name, text in readmes.items():
                if shown(c['value'], c['digits'], comma=name.endswith('.it.md')) not in text:
                    problems.append(f"{c['key']}: {shown(c['value'], c['digits'], name.endswith('.it.md'))} not found in {name}")
    return problems


def main():
    claims = json.loads((ROOT / 'published_results' / 'claims.json').read_text(encoding='utf-8'))
    problems = verify(claims)
    for p in problems:
        print(p)
    print(f'{len(problems)} problem(s) in {len(claims)} claims' if problems else f'ok: {len(claims)} claims match')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
