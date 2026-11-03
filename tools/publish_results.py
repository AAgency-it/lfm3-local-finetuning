"""Copy result summaries into published_results/, numbers only, with private names turned into neutral labels.

    python tools/publish_results.py <results_dir> [pattern ...]

- Per-case detail (`detail`, `dettaglio`, `previsioni`, per-sentence scores) is dropped: only aggregate numbers are published.
- Names that must not be public (brands, customers) are replaced with labels such as "set A", using the git-ignored
  tools/label_map.local.json ({"name": "label"}). The labels are what you see in the README and in the figures.
- Internal names of our own tools are renamed with the git-ignored tools/rename.local.json ([["internal", "public"], ...]).
- If a word from tools/forbidden_words.txt survives the relabelling, the file is NOT written and the script fails.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'published_results'
DROP = {'detail', 'dettaglio', 'previsioni', 'per_frase', 'casi'}
# keys only: in free text "brand" is an ordinary word
KEY_RENAME = [('brand', 'category')]


def _rename(s, pairs):
    for old, new in pairs:
        s = s.replace(old, new)
    return s


def _relabel(s, labels):
    """Replace whole words only, so a short code such as "NEW" never eats the inside of a longer name."""
    for name, label in labels.items():
        s = re.sub(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', label, s, flags=re.I)
    return s


def anonymise(obj, labels, forbidden=(), rename=()):
    """rename: ordered (internal, public) pairs applied inside keys and strings, from tools/rename.local.json."""
    def walk(x):
        if isinstance(x, dict):
            return {_rename(_rename(_relabel(k, labels), rename), KEY_RENAME): walk(v) for k, v in x.items()
                    if k not in DROP or isinstance(v, (int, float))}  # a count is kept, the per-case list is not
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            return _rename(_relabel(x, labels), rename)
        return x
    out = walk(obj)
    text = json.dumps(out, ensure_ascii=False).lower()
    left = [w for w in forbidden if w in text]
    if left:
        raise ValueError(f'forbidden words left after relabelling: {left}')
    return out


def _local(name, default):
    p = ROOT / 'tools' / name
    if not p.exists():
        return default
    if p.suffix == '.json':
        return json.loads(p.read_text(encoding='utf-8'))
    return [w.strip().lower() for w in p.read_text(encoding='utf-8').splitlines() if w.strip()]


def main():
    src = sys.argv[1]
    patterns = sys.argv[2:] or ['*.json']
    labels = _local('label_map.local.json', {})
    forbidden = _local('forbidden_words.txt', [])
    rename = _local('rename.local.json', [])
    DEST.mkdir(exist_ok=True)
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(src, pat))):
            out = anonymise(json.load(open(f, encoding='utf-8')), labels, forbidden, rename)
            name = _relabel(os.path.basename(f), labels)
            (DEST / name).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
            print(name)


if __name__ == '__main__':
    main()
