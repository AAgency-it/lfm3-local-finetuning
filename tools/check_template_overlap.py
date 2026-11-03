"""How many sentences of the extraction bench follow a sentence template of the training-data generator.

    python tools/check_template_overlap.py           (prints the count and the ids)
    python tools/check_template_overlap.py --write   (also writes published_results/extract_template_overlap.json)

A bench sentence "follows" a template when it is the template with any text in place of {D} (date), {A} (amount) and {n}
(number). Those sentences share their wording with the training data, so on them the fine-tuned model's score is a ceiling.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data_synthetic'))
import gen_extract  # noqa: E402

TEMPLATES = gen_extract.MODELLI_ENTRAMBI + gen_extract.MODELLI_DATA + gen_extract.MODELLI_IMPORTO + gen_extract.MODELLI_NULLA


def pattern(template):
    parts = re.split(r'(\{[DAn]\})', template)
    return re.compile('^' + ''.join('.+?' if p in ('{D}', '{A}', '{n}') else re.escape(p) for p in parts) + '$', re.I)


def follows(template, sentence):
    return bool(pattern(template).match(sentence.strip()))


def overlapping(bench=ROOT / 'bench' / 'extract_bench.jsonl'):
    """Ids of the bench sentences that follow one of the generator's templates."""
    compiled = [pattern(t) for t in TEMPLATES]
    ids = []
    for line in bench.read_text(encoding='utf-8').splitlines():
        if line.strip():
            case = json.loads(line)
            if any(p.match(case['testo'].strip()) for p in compiled):
                ids.append(case['id'])
    return ids


def main(argv):
    ids = overlapping()
    total = sum(1 for line in (ROOT / 'bench' / 'extract_bench.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
    print(f'{len(ids)} of {total} bench sentences follow one of {len(TEMPLATES)} generator templates: {" ".join(ids)}')
    if '--write' in argv:
        out = {'bench_sentences': total, 'templates': len(TEMPLATES), 'following_a_template': len(ids), 'ids': ids}
        (ROOT / 'published_results' / 'extract_template_overlap.json').write_text(
            json.dumps(out, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
