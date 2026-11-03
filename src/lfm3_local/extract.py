"""Skill `extract`: from one Italian sentence, pull the date (YYYY-MM-DD) and the amount in euro, or null.

Order of work, and the reason it is built this way:
 1. Regexes first (`by_regex`): if the sentence has at most one complete date and one amount written in digits, answer with no model.
 2. Only ambiguous sentences go to the model, constrained by a JSON schema.
 3. The model's answer passes a validator written in plain code: it is accepted only if the date and the amount are really written
    in the sentence. Rejected = the caller decides (a bigger model, or a human).

The model proposes, the code disposes. On our 50-sentence bench the regexes alone get 45 right; the model adds a few hard cases.
Italian-language patterns; adapt `MONTHS` and the patterns for other languages.
"""
import json
import re
from datetime import date

from . import client

NAME = 'extract'
MONTHS = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
SYSTEM = ("Estrai da una frase la data e l'importo in euro. Data in formato AAAA-MM-GG, importo come numero "
          "con il punto decimale (es. 1250.5). Se la frase non contiene una data o un importo in euro, metti null. "
          "Le percentuali non sono importi.")
SCHEMA = {'type': 'object', 'properties': {
    'data': {'type': ['string', 'null'], 'description': 'date as YYYY-MM-DD, null if absent'},
    'importo_euro': {'type': ['number', 'null'], 'description': 'amount in euro as a number, null if absent'}},
    'required': ['data', 'importo_euro'], 'additionalProperties': False}

_NUM = r'(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)'
_EURO_AFTER = re.compile(_NUM + r'\s*(?:€|euro|eur\b)', re.I)
_EURO_BEFORE = re.compile(r'€\s*' + _NUM)
_DATE_NUMERIC = re.compile(r'(?<![\d/.\-])(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})(?![\d/\-]|\.\d)')
_DATE_ISO = re.compile(r'(?<![\d\-])(\d{4})-(\d{2})-(\d{2})(?![\d\-])')
_DATE_TEXT = re.compile(r'(?<!\d)(\d{1,2})\s*°?\s+(' + '|'.join(MONTHS) + '|' + '|'.join(m[:3] for m in MONTHS) + r')\b\.?\s+(\d{4})', re.I)
_EURO_WORD = re.compile(r'€|\beuro\b|\beur\b', re.I)
_RANGE = re.compile(r'\b(?:da|tra|fra)\s+[\d.,]+\s*(?:€|euro)?\s+(?:a|e)\s+[\d.,]+', re.I)  # "from 120 to 135 euro": which amount?


def _number(s):
    return float(s.replace('.', '').replace(',', '.'))


def amounts_in_text(text):
    """Every euro amount written in the sentence (number followed by euro/€, or preceded by €)."""
    return sorted({_number(m.group(1)) for m in _EURO_AFTER.finditer(text)} | {_number(m.group(1)) for m in _EURO_BEFORE.finditer(text)})


def dates_in_text(text):
    """Every complete date (day, month and year) written in the sentence, as YYYY-MM-DD strings."""
    found = set()
    for d, m, y in _DATE_NUMERIC.findall(text):
        year = int(y) + (2000 if len(y) == 2 else 0)
        try:
            found.add(date(year, int(m), int(d)).isoformat())
        except ValueError:
            pass
    for y, m, d in _DATE_ISO.findall(text):
        try:
            found.add(date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            pass
    for d, month, y in _DATE_TEXT.findall(text):
        try:
            found.add(date(int(y), [m[:3] for m in MONTHS].index(month.lower()[:3]) + 1, int(d)).isoformat())
        except ValueError:
            pass
    return sorted(found)


def validate(text, r):
    """(ok, reasons). `r` is the dict proposed by the model."""
    reasons = []
    if not isinstance(r, dict) or set(r) != {'data', 'importo_euro'}:
        return False, ['format']
    amounts, dates = amounts_in_text(text), dates_in_text(text)
    d, amount = r['data'], r['importo_euro']
    if d is None:
        if dates:
            reasons.append('date present in the sentence but null')
    elif d not in dates:
        reasons.append('date not written in the sentence')
    if amount is None:
        if amounts:
            reasons.append('amount present in the sentence but null')
    elif not isinstance(amount, (int, float)) or not any(abs(amount - x) < 0.005 for x in amounts):
        reasons.append('amount not written in euro in the sentence')
    return not reasons, reasons


def by_regex(text):
    """Answer without a model, only when there is no ambiguity: at most one date and one amount in digits, no 'euro' without digits
    (amount in words), no range. Otherwise None and the model decides, under the validator."""
    d, a = dates_in_text(text), amounts_in_text(text)
    if len(d) > 1 or len(a) > 1 or (_EURO_WORD.search(text) and not a) or _RANGE.search(text):
        return None
    return {'data': d[0] if d else None, 'importo_euro': a[0] if a else None}


def run(text):
    """`outcome`: 'accepted' (regex, or model that passed the validator) or 'escalate' (rejected, or server off)."""
    r = by_regex(text)
    if r is not None:
        client.log(NAME, text, 'accepted', 0, {'source': 'regex'})
        return {'outcome': 'accepted', **r, 'source': 'regex', 'ms': 0}
    answer = client.ask(SYSTEM, 'Frase: ' + text, schema=SCHEMA, max_tokens=60, temperature=0.0)
    if not answer['ok']:
        client.log(NAME, text, 'escalate', 0, {'reason': answer['reason']})
        return {'outcome': 'escalate', 'reason': answer['reason']}
    try:
        r = json.loads(answer['text'])
    except ValueError:
        r = None
    ok, reasons = validate(text, r)
    client.log(NAME, text, 'accepted' if ok else 'escalate', answer['ms'], {'reasons': reasons})
    if ok:
        return {'outcome': 'accepted', **r, 'source': 'model', 'ms': answer['ms']}
    return {'outcome': 'escalate', 'reason': '; '.join(reasons), 'proposal': r, 'ms': answer['ms']}
