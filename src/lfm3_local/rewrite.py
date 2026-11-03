"""Skill `rewrite`: rewrite one short sentence in direct Italian, without redundant courtesy formulas.

The result is a DRAFT, never finished text. The validator only checks what code can check: shorter, same numbers, no new numbers,
finished sentence, one line. The meaning (a request must stay a request, a future must stay a future) is NOT verified.
In our hand check, 3 of 9 accepted rewrites changed the meaning (docs/results.md). A human reads the draft.
"""
import re

from . import client

NAME = 'rewrite'
MAX_CHARS = 600  # past this the model repeats the instruction or invents
SYSTEM = ("Riscrivi la frase in italiano diretto: breve, senza formule di cortesia ridondanti, senza giri di parole. "
          "Mantieni tutti i fatti (date, numeri, nomi, azioni) e non aggiungere nulla. Rispondi solo con la frase riscritta.")


def numbers(text):
    return sorted(re.findall(r'\d+(?:[.,]\d+)*', text))


def validate(original, rewritten):
    reasons = []
    if not rewritten.strip():
        reasons.append('empty')
    else:
        if len(rewritten) >= len(original):
            reasons.append('not shorter')
        if rewritten.strip().lower() == original.strip().lower():
            reasons.append('identical')
        if numbers(rewritten) != numbers(original):
            reasons.append('numbers differ from the original')
        if rewritten.rstrip()[-1] not in '.!?':
            reasons.append('sentence not finished')
        if '\n' in rewritten.strip() or '<' in rewritten:
            reasons.append('more than one line or markup')
    return not reasons, reasons


def run(text):
    if len(text) > MAX_CHARS:
        return {'outcome': 'escalate', 'reason': f'text over {MAX_CHARS} characters'}
    answer = client.ask(SYSTEM, text, max_tokens=160, temperature=0.2)
    if not answer['ok']:
        client.log(NAME, text, 'escalate', 0, {'reason': answer['reason']})
        return {'outcome': 'escalate', 'reason': answer['reason']}
    ok, reasons = validate(text, answer['text'])
    client.log(NAME, text, 'draft' if ok else 'escalate', answer['ms'], {'reasons': reasons})
    if ok:
        return {'outcome': 'draft', 'text': answer['text'], 'ms': answer['ms'], 'warning': 'draft: numbers and length checked, meaning not'}
    return {'outcome': 'escalate', 'reason': '; '.join(reasons), 'proposal': answer['text'], 'ms': answer['ms']}
