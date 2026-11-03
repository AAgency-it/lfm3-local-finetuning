"""Single client for LFM3 1.7B served locally by llama-server (OpenAI-compatible API, default http://127.0.0.1:8018).

Standard library only. It does not start the server. Every call returns a dict:
  {'ok': True, 'text': ..., 'ms': ..., 'tokens': ...}   or   {'ok': False, 'reason': 'server' | 'too_long'}
No exception leaves this module: if the server is off or answers badly, the caller falls back to a bigger model or a human.

Thinking is always switched off (enable_thinking=False). Without it LFM3 spends its token budget reasoning and the
visible answer comes back empty. See docs/pitfalls.md.
"""
import hashlib
import json
import math
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

MAX_TEXT = 4000  # hard cap in characters; past ~500 the 1.7B model starts repeating the instruction or inventing
# Sampling parameters suggested on the model card: temperature 0.1, top_k 50, repetition penalty 1.1.
PARAMS = {'temperature': 0.1, 'top_k': 50, 'repeat_penalty': 1.1}


def server():
    return os.environ.get('LFM_SERVER', 'http://127.0.0.1:8018')


def _log_dir():
    return Path(os.environ.get('LFM_LOG', 'lfm-log'))


def fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _post(path, body=None, timeout=60):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(server() + path, data=data, headers={'Content-Type': 'application/json; charset=utf-8'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def healthy():
    """True if the server answers {"status": "ok"}. Never raises."""
    try:
        return _post('/health', timeout=2).get('status') == 'ok'
    except Exception:
        return False


def log(skill, text, outcome, ms, extra=None):
    """Append one line to the monthly JSONL log: skill, fingerprint of the text, outcome, time. Never the text itself."""
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        row = {'when': f'{datetime.now():%Y-%m-%dT%H:%M:%S}', 'skill': skill, 'fingerprint': fingerprint(text),
               'outcome': outcome, 'ms': ms, **(extra or {})}
        with open(d / f'{datetime.now():%Y-%m}.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError:
        pass  # logging must never stop the work


def ask(system, user, schema=None, grammar=None, max_tokens=200, temperature=None, logprobs=False):
    """One request, thinking off. `schema` = JSON schema, `grammar` = GBNF. Refuses more than MAX_TEXT characters."""
    if len(user) > MAX_TEXT:
        return {'ok': False, 'reason': 'too_long', 'detail': f'{len(user)} characters, max {MAX_TEXT}'}
    body = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False}, **PARAMS}
    if temperature is not None:
        body['temperature'] = temperature
    if schema is not None:
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'r', 'schema': schema}}
    if grammar is not None:
        body['grammar'] = grammar
    if logprobs:
        body['logprobs'] = True
        body['top_logprobs'] = 20
    t0 = time.perf_counter()
    try:
        d = _post('/v1/chat/completions', body, timeout=120)
        text = d['choices'][0]['message']['content'] or ''
    except Exception as e:  # off, busy, malformed
        return {'ok': False, 'reason': 'server', 'detail': type(e).__name__}
    ms = round((time.perf_counter() - t0) * 1000)
    return {'ok': True, 'text': text.split('</think>')[-1].strip(), 'ms': ms,
            'tokens': d.get('usage', {}).get('completion_tokens'), 'raw': d}


def first_choice_probability(raw, options):
    """Probability of the first token among `options` (enum grammar). None if two options share their first token
    or the log-probabilities are missing. Useful as a weak signal; see docs/results.md for why it is not a safety filter."""
    try:
        toks = raw['choices'][0]['logprobs']['content']
        top = toks[0]['top_logprobs']
        chosen = toks[0]['token'].strip()
    except (KeyError, IndexError, TypeError):
        return None
    weights = {}
    for t in top:
        word = t['token'].strip()
        if not word:
            continue
        for o in options:
            if o.startswith(word):
                weights[o] = weights.get(o, 0.0) + math.exp(t['logprob'])
    prefixed = [o for o in options if o.startswith(chosen)]
    if len(prefixed) != 1 or not weights:
        return None
    total = sum(weights.values())
    return round(weights.get(prefixed[0], 0.0) / total, 3) if total else None
