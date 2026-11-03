"""Pre-commit cleanliness check: fails (exit 1) if the repo contains things that must not be published.

Looks for: absolute personal paths, user names, drive-letter paths, tokens/keys/passwords, .env files, e-mail addresses, private IPs,
and a configurable list of words that must never appear (customer/brand/people names, internal rules) read from
tools/forbidden_words.txt if you create it (one word per line, case-insensitive; it is git-ignored on purpose).
Every text file is scanned, JSON results and SVG figures included.

    python tools/check_clean.py              the working tree, skipping .git
    python tools/check_clean.py --history    also every line ever added in any commit of any branch (git log -p --all)
Run it before every commit. A green run is necessary, not sufficient: read the diff too.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {'.git', '__pycache__', 'results', 'data', '.venv', '.pytest_cache'}
SKIP_FILES = {'check_clean.py', 'forbidden_words.txt', 'label_map.local.json', 'rename.local.json'}
PATTERNS = {
    'windows path': re.compile(r'\b[A-Za-z]:[\\/](?!\\)[^\s"\']{2,}'),
    'user home path': re.compile(r'(?:/Users/|/home/|\\Users\\)[A-Za-z0-9._-]+', re.I),
    'secret-like': re.compile(r'(?i)\b(api[_-]?key|secret|token|passwd|password|bearer)\b\s*[:=]\s*[\'"]?[A-Za-z0-9_\-]{12,}'),
    'private key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    'hf/kaggle token': re.compile(r'\b(hf_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]+)\b'),
    'e-mail': re.compile(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b'),  # the domain ends in letters: package@1.2.3 is not an address
    'private ip': re.compile(r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b'),
}
ALLOWED_EMAIL = re.compile(r'noreply@|@example\.')


def forbidden_words():
    fw = ROOT / 'tools' / 'forbidden_words.txt'
    return [w.strip().lower() for w in fw.read_text(encoding='utf-8').splitlines() if w.strip()] if fw.exists() else []


def scan_text(text, words, where=''):
    """Return (where:line, label, hit) for every problem found in text."""
    problems = []
    for n, line in enumerate(text.splitlines(), 1):
        for label, rx in PATTERNS.items():
            for m in rx.finditer(line):
                if label == 'e-mail' and ALLOWED_EMAIL.search(m.group(0)):
                    continue
                problems.append((f'{where}:{n}', label, m.group(0)[:60]))
        low = line.lower()
        problems += [(f'{where}:{n}', 'forbidden word', w) for w in words if w in low]
    return problems


def scan_tree(root, words):
    problems = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if not p.is_file() or p.name in SKIP_FILES or any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.name == '.env' or p.suffix == '.env':
            problems.append((str(rel), 'env file', p.name))
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        problems += scan_text(text, words, str(rel))
    return problems


def scan_history(root, words):
    """Scan every added line of every commit (content only: authors are checked separately with git log --format)."""
    out = subprocess.run(['git', '-C', str(root), 'log', '-p', '--all', '--format=commit %h', '--no-color'],
                         capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
    commit, path, added = '?', '?', []
    for line in out.splitlines():
        if line.startswith('commit '):
            commit = line.split()[1]
        elif line.startswith('+++ '):
            path = line[6:] if line.startswith('+++ b/') else line[4:]
        elif line.startswith('+') and not line.startswith('+++') and Path(path).name not in SKIP_FILES:
            added.append((f'{commit}:{path}', line[1:]))
    problems = []
    for where, text in added:
        problems += [(where, label, hit) for _, label, hit in scan_text(text, words)]
    return problems


def main():
    words = forbidden_words()
    problems = scan_tree(ROOT, words)
    if '--history' in sys.argv:
        problems += scan_history(ROOT, words)
    for where, label, hit in problems:
        print(f'{where}: {label}: {hit}')
    print(f'{len(problems)} problem(s)' if problems else 'clean')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
