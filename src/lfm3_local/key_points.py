"""Skill `key_points`: from a long text (minutes, email, chat, report) pick the sentences that matter, copied as written.

It does not summarise or rewrite. It judges each sentence (decision, commitment with owner and/or deadline, agreed date or amount,
blocker) and returns ONLY sentences that are in the text, with their position. It cannot invent: at worst it skips one
(recall is never 1.0), so it is a filter to find your way, not a replacement for reading when the decision hinges on a detail.

Why per-sentence and not "summarise this": the 1.7B model summarised long texts badly (1 of 10 acceptable in one-line summaries,
4 of 10 with a better prompt). One short yes/no question per sentence, constrained by a grammar, is what it can do.
The threshold is chosen on a development set, never on the test set.

This is the skill we tried to improve with fine-tuning: see docs/finetuning.md and docs/results.md.
"""
import re

from . import client

NAME = 'key_points'
THRESHOLD = 0.5
MAX_SENTENCES = 120
SYSTEM = ("Decidi se la frase TARGET di un testo è una frase chiave. È chiave solo se contiene almeno una di queste cose: "
          "una decisione presa, un impegno con chi fa che cosa e/o entro quando, una data o un importo in euro concordati, "
          "un problema che blocca il lavoro. Non è chiave: saluti, contesto, opinioni, ipotesi, fatti già successi senza impegno, "
          "date o importi solo ricordati. Rispondi solo «chiave» oppure «altro».")
OPTIONS = ['chiave', 'altro']
GRAMMAR = 'root ::= "chiave" | "altro"'

_END = re.compile(r'(?<=[.!?…])\s+')


def split_sentences(text):
    """One sentence per line, long lines split at the full stop."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out += [s.strip() for s in _END.split(line) if s.strip()]
    return out


def sentence_prompt(sentences, i):
    before = sentences[i - 1] if i > 0 else ''
    after = sentences[i + 1] if i + 1 < len(sentences) else ''
    return f'PRIMA: {before}\nTARGET: {sentences[i]}\nDOPO: {after}'


def score(sentences, i, system=SYSTEM):
    """Probability that sentence i is a key one, or None if the server does not answer."""
    r = client.ask(system, sentence_prompt(sentences, i), grammar=GRAMMAR, max_tokens=4, temperature=0.0, logprobs=True)
    if not r['ok']:
        return None
    p = client.first_choice_probability(r['raw'], OPTIONS)
    if p is None:
        return None
    return p if r['text'].strip() == 'chiave' else 1 - p


def run(text, threshold=THRESHOLD):
    """`outcome`: 'extracted' (key sentences with position) or 'escalate' (empty, too long, server off)."""
    sentences = split_sentences(text)
    if not sentences or len(sentences) > MAX_SENTENCES:
        return {'outcome': 'escalate', 'reason': f'need 1 to {MAX_SENTENCES} sentences (found {len(sentences)})'}
    key, scores = [], []
    for i in range(len(sentences)):
        p = score(sentences, i)
        if p is None:
            client.log(NAME, text, 'escalate', 0, {'reason': 'server'})
            return {'outcome': 'escalate', 'reason': 'server'}
        scores.append(round(p, 3))
        if p >= threshold:
            key.append(i)
    client.log(NAME, text, 'extracted', 0, {'sentences': len(sentences), 'key': len(key), 'threshold': threshold})
    return {'outcome': 'extracted', 'total_sentences': len(sentences), 'key': [{'position': i, 'sentence': sentences[i]} for i in key],
            'scores': scores, 'warning': 'a filter to find your way: it can skip a sentence; read the text if a detail decides'}
