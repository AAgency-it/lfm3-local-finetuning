"""Two generic skills: `classify` (pick one of a few labels) and `in_pieces` (rewrite ONE piece of code, with a check).

`classify` is deliberately small. It returns a label only when the first-token probability is above a threshold, and it accepts an
optional `deny` regex: put there the topics you never want a 1.7B model to decide for you (legal, medical, money, approvals).
On our own tests the base model was weak on policy-style yes/no decisions (0.537 on the rule checks, against 0.50 for always giving
the same answer) and its confidence did not help (see docs/results.md): treat the label as a suggestion, never as a gate.

`in_pieces` is the method that worked: a bigger model writes the skeleton and says what the piece must contain, LFM writes that
piece only, a deterministic check accepts it or sends it back (up to 3 attempts).
"""
import re

from . import client

MAX_OPTIONS = 12


def classify(instruction, text, options, threshold=0.8, deny=None):
    """One label among `options`. `outcome`: 'proposal' (probability >= threshold) or 'escalate'."""
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS or not all(isinstance(o, str) and o.strip() for o in options):
        return {'outcome': 'escalate', 'reason': f'need 2 to {MAX_OPTIONS} text labels'}
    if deny is not None and (re.search(deny, instruction, re.I) or any(re.search(deny, o, re.I) for o in options)):
        return {'outcome': 'escalate', 'reason': 'topic excluded by the deny pattern'}
    grammar = 'root ::= ' + ' | '.join('"%s"' % o.replace('"', '') for o in options)
    system = f'{instruction.strip()}\nAnswer with one of these labels only: {", ".join(options)}.'
    r = client.ask(system, text, grammar=grammar, max_tokens=12, temperature=0.0, logprobs=True)
    if not r['ok']:
        client.log('classify', text, 'escalate', 0, {'reason': r['reason']})
        return {'outcome': 'escalate', 'reason': r['reason']}
    choice = r['text'].strip()
    p = client.first_choice_probability(r['raw'], options)
    ok = choice in options and p is not None and p >= threshold
    client.log('classify', text, 'proposal' if ok else 'escalate', r['ms'], {'choice': choice, 'p': p})
    out = {'outcome': 'proposal' if ok else 'escalate', 'choice': choice if choice in options else None, 'probability': p, 'ms': r['ms']}
    if not ok:
        out['reason'] = 'probability below threshold or not computable (labels share their first token)' if choice in options else 'label not in the list'
    return out


def check_piece(piece, must_contain, must_not_contain, starts_with):
    reasons = []
    if not piece.strip():
        return ['empty']
    if '```' in piece:
        reasons.append('contains ```')
    if starts_with and not piece.lstrip().startswith(starts_with):
        reasons.append(f'does not start with "{starts_with}"')
    reasons += [f'missing "{s}"' for s in must_contain if s not in piece]
    reasons += [f'contains "{s}"' for s in must_not_contain if s in piece]
    for a, b in ('{}', '()', '[]'):
        if piece.count(a) != piece.count(b):
            reasons.append(f'{a}{b} unbalanced')
    return reasons


def in_pieces(instruction, current_code='', must_contain=(), must_not_contain=(), starts_with=None, max_tokens=400, attempts=3):
    """Rewrite one piece. Returns the first attempt that passes the check, else 'escalate' with the last attempt."""
    system = 'Answer with the requested code only, no explanations and no ``` fences.'
    user = instruction.strip() + (f'\n\nCurrent code:\n{current_code}' if current_code else '')
    last, reasons, tokens = None, [], 0
    for n in range(1, max(1, min(int(attempts), 3)) + 1):
        r = client.ask(system, user, max_tokens=int(max_tokens), temperature=0.1 if n == 1 else 0.4)
        if not r['ok']:
            client.log('in_pieces', user, 'escalate', 0, {'reason': r['reason']})
            return {'outcome': 'escalate', 'reason': r['reason']}
        tokens += r.get('tokens') or 0
        last = re.sub(r'^```\w*\n|\n```$', '', r['text']).strip()
        reasons = check_piece(last, list(must_contain), list(must_not_contain), starts_with)
        if not reasons:
            client.log('in_pieces', user, 'accepted', r['ms'], {'attempts': n, 'tokens': tokens})
            return {'outcome': 'accepted', 'piece': last, 'attempts': n, 'tokens': tokens}
    client.log('in_pieces', user, 'escalate', 0, {'attempts': attempts, 'reasons': reasons})
    return {'outcome': 'escalate', 'reason': '; '.join(reasons), 'last_attempt': last, 'tokens': tokens}
