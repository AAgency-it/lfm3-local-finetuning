"""Draw the README figures as plain SVG from the published numbers, so a figure can never disagree with its file.

    python figures/make_figures.py        (writes docs/img/*.svg; run it again and git diff must stay empty)

Numbers come from published_results/claims.json (the same list check_numbers.py verifies against the READMEs) and from
published_results/hardware_f1_base_tre_avvii.json. The terminal figure renders figures/terminal.txt, a transcript of real
commands run in this repository. No randomness, no timestamps. Colours: three categorical slots validated for colour-blind
separation on the light surface; every bar also carries its value as text, so colour is never the only cue.
"""
import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'img'
W = 720
SURFACE, BORDER, INK, INK2, GRID = '#FCFCFA', '#D0D7DE', '#1F2328', '#59636E', '#E4E7EB'
BASE, TUNED, NOMODEL = '#2a78d6', '#eb6834', '#1baf7a'
FONT = '-apple-system, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, &quot;Cascadia Mono&quot;, Consolas, Menlo, monospace'


def claims():
    return {c['key']: c['value'] for c in json.loads((ROOT / 'published_results' / 'claims.json').read_text(encoding='utf-8'))}


def svg(height, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img" '
            f'aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            f'<style>text {{ font-family: {FONT}; fill: {INK}; }} .s {{ fill: {INK2}; }} .m {{ font-family: {MONO}; }}</style>\n'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" fill="{SURFACE}" stroke="{BORDER}"/>\n'
            + body + '</svg>\n')


def t(x, y, s, size=20, weight=400, cls='', anchor='start'):
    c = f' class="{cls}"' if cls else ''
    a = f' text-anchor="{anchor}"' if anchor != 'start' else ''
    w = f' font-weight="{weight}"' if weight != 400 else ''
    return f'<text x="{x}" y="{y}" font-size="{size}"{w}{c}{a}>{escape(s)}</text>\n'


def num(v, digits=3):
    return f'{v:.{digits}f}'


def legend(y, items):
    out, x = '', 40
    for label, colour in items:
        out += f'<rect x="{x}" y="{y - 15}" width="18" height="18" rx="4" fill="{colour}"/>\n' + t(x + 26, y, label, 20, cls='s')
        x += 40 + int(len(label) * 10.5)
    return out


def bars(groups, top, scale_w=440, x0=40):
    """groups: [(title, [(series_label, value, colour, digits), ...])]. Horizontal bars on a 0-1 scale, value written after each bar."""
    out, y = '', top
    for title, rows in groups:
        out += t(x0, y, title, 21, 600)
        y += 14
        for _, value, colour, digits in rows:
            w = max(4, round(value * scale_w))
            out += f'<rect x="{x0}" y="{y}" width="{w}" height="26" rx="4" fill="{colour}"/>\n'
            out += t(x0 + w + 10, y + 20, num(value, digits), 20, 600)
            y += 26 + 4
        y += 34
    return out, y


def three_numbers(c):
    groups = [
        ('Rule checks (320 sentences, accuracy)', [('base', c['claim_base'], BASE, 3), ('fine-tuned', c['claim_multi'], TUNED, 3),
                                                    ('no model', c['claim_majority'], NOMODEL, 2)]),
        ('Key sentences (60 synthetic docs, F1)', [('base', c['kp_base'], BASE, 3), ('fine-tuned', c['kp_round1'], TUNED, 3),
                                                   ('no model', c['kp_keywords'], NOMODEL, 3)]),
        ('Date + amount (50 sentences, exact)', [('base', c['extract_base'], BASE, 2), ('fine-tuned', c['extract_multi'], TUNED, 2),
                                                 ('no model', c['extract_regex'], NOMODEL, 2)]),
    ]
    head = t(40, 52, 'Three numbers, base vs fine-tuned', 30, 700) + t(40, 84, 'Scale 0 to 1. Fine-tuned numbers are ceilings: see the README.', 20, cls='s')
    head += legend(124, [('base LFM3 1.7B', BASE), ('fine-tuned', TUNED), ('no model', NOMODEL)])
    body, y = bars(groups, 176)
    desc = ' '.join(f'{g}: ' + ', '.join(f'{lab} {num(v, d)}' for lab, v, _, d in rows) + '.' for g, rows in groups)
    return svg(y + 6, 'Three numbers: base model, fine-tuned model, no model', desc, head + body)


def where_it_loses(c):
    groups = [
        ('Task category from the title (accuracy)', [('fine-tuned', c['category_multi'], TUNED, 3),
                                                      ('label already written by hand', c['category_existing_label'], NOMODEL, 3)]),
        ('Key sentences in real notes (F1)', [('base', c['notes_kp_base'], BASE, 3), ('fine-tuned', c['notes_kp_multi'], TUNED, 3),
                                              ('keyword list', c['notes_kp_keywords'], NOMODEL, 3)]),
        ('Real log lines, relevant or noise (accuracy)', [('base', c['logs_base'], BASE, 3), ('fine-tuned', c['logs_multi'], TUNED, 3),
                                                          ('"everything is noise"', c['logs_majority'], NOMODEL, 3)]),
    ]
    head = t(40, 52, 'Where it loses, or wins only a little', 30, 700)
    head += t(40, 84, f'Real material, not synthetic. On real notes precision is only {num(c["notes_kp_multi_precision"])}.', 20, cls='s')
    head += legend(124, [('base', BASE), ('fine-tuned', TUNED), ('no model / existing label', NOMODEL)])
    body, y = bars(groups, 176)
    desc = ' '.join(f'{g}: ' + ', '.join(f'{lab} {num(v, d)}' for lab, v, _, d in rows) + '.' for g, rows in groups)
    return svg(y + 6, 'Where the fine-tuned model loses or wins only a little', desc, head + body)


def box(x, y, w, h, lines, fill='#FFFFFF', stroke=INK2, bold_first=True):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>\n'
    yy = y + 34
    for i, line in enumerate(lines):
        out += t(x + w // 2, yy, line, 21 if i == 0 else 19, 700 if (i == 0 and bold_first) else 400, '' if i == 0 else 's', 'middle')
        yy += 28
    return out


def arrow(x1, y1, x2, y2, label='', lx=None, ly=None):
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="2" marker-end="url(#ah)"/>\n'
    if label:
        out += t(lx if lx is not None else (x1 + x2) // 2 + 10, ly if ly is not None else (y1 + y2) // 2 + 6, label, 19, cls='s')
    return out


def orchestrator():
    defs = (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n')
    b = defs + t(40, 52, 'The small model works under a bigger one', 30, 700)
    b += t(40, 84, 'It never decides alone: a code check accepts or rejects every answer.', 20, cls='s')
    b += box(200, 112, 320, 92, ['Bigger model (Claude)', 'gives a narrow job'])
    b += arrow(360, 204, 360, 252, 'prompt + schema')
    b += box(200, 254, 320, 92, ['LFM3 1.7B, local', '6 GB card, no API bill'], fill='#FDF0EA', stroke=TUNED)
    b += arrow(360, 346, 360, 394, 'answer')
    b += box(200, 396, 320, 92, ['Check in code', 'schema, values, rules'])
    b += arrow(360, 488, 360, 536, 'passes')
    b += box(200, 538, 320, 64, ['Answer used'], bold_first=True)
    b += f'<path d="M520 442 H620 V158 H528" fill="none" stroke="{INK2}" stroke-width="2" marker-end="url(#ah)"/>\n'
    b += t(632, 290, 'fails:', 19, cls='s') + t(632, 316, 'back to', 19, cls='s') + t(632, 342, 'the big', 19, cls='s') + t(632, 368, 'model', 19, cls='s')
    b += t(40, 646, 'Every call is logged: what was asked, the outcome, and whether it went back.', 19, cls='s')
    desc = ('A bigger model (Claude) gives a narrow job with a prompt and a schema to LFM3 1.7B running locally on a 6 GB card. '
            'A check in code validates the answer: if it passes the answer is used, if it fails the job goes back to the bigger model. '
            'Every call is logged.')
    return svg(676, 'How the small model works under a bigger one', desc, b)


def hardware():
    h = json.loads((ROOT / 'published_results' / 'hardware_f1_base_tre_avvii.json').read_text(encoding='utf-8'))
    total, used = h['vram_totale_mib'], h['vram_server_mib'][0]
    b = t(40, 52, 'One gaming card is enough to run it', 30, 700)
    b += t(40, 84, 'GTX 1660 Ti, base LFM3 1.7B (Q4_K_M), measured 2026-11-03.', 20, cls='s')
    b += t(40, 136, 'GPU memory taken by the server', 21, 600)
    bw = W - 80
    uw = round(bw * used / total)
    b += f'<rect x="40" y="152" width="{bw}" height="30" rx="4" fill="{GRID}"/>\n'
    b += f'<rect x="40" y="152" width="{uw}" height="30" rx="4" fill="{BASE}"/>\n'
    b += t(40, 212, f'{used:,} MiB of {total:,} MiB ({round(100 * used / total)}%)', 20, 600)
    b += t(40, 270, 'Generation speed, three separate starts', 21, 600)
    lo, hi = 0, 140
    ax0, ax1, ay = 40, W - 60, 330
    b += f'<line x1="{ax0}" y1="{ay}" x2="{ax1}" y2="{ay}" stroke="{BORDER}" stroke-width="2"/>\n'
    for v in (0, 20, 40, 60, 80, 100, 120, 140):
        x = ax0 + (ax1 - ax0) * (v - lo) / (hi - lo)
        b += f'<line x1="{x:.1f}" y1="{ay - 6}" x2="{x:.1f}" y2="{ay + 6}" stroke="{BORDER}" stroke-width="2"/>\n'
        b += t(round(x), ay + 32, str(v), 18, cls='s', anchor='middle')
    for v in sorted(h['tok_s_generazione_mediane']):
        x = ax0 + (ax1 - ax0) * (v - lo) / (hi - lo)
        b += f'<circle cx="{x:.1f}" cy="{ay}" r="9" fill="{BASE}" stroke="{SURFACE}" stroke-width="2"/>\n'
    vals = ', '.join(f'{v:.1f}' for v in h['tok_s_generazione_mediane'])
    b += t(40, ay + 74, f'tokens per second: {vals}', 20, 600)
    b += t(40, ay + 104, 'Same machine and prompt: the speed changes from one start to the next.', 19, cls='s')
    desc = (f'The server takes {used} MiB of the {total} MiB card. Generation speed in three separate starts: {vals} tokens per second. '
            'The fine-tuned Q8_0 models were not measured here.')
    return svg(ay + 134, 'GPU memory and speed of the base model on a GTX 1660 Ti', desc, b)


def timeline(c):
    pts = [('2026-10-30', 'Base model', 'runs locally'), ('2026-10-30', 'Run 1', f'key sentences, F1 {num(c["kp_round1"])}'),
           ('2026-10-31', 'Run 2', f'{int(c["round2_rows"]):,} rows, 5 tasks'), ('2026-11-03', 'Generic adapter', 'synthetic only, bench 0.994')]
    b = t(40, 52, 'Three runs on a free GPU', 30, 700)
    b += t(40, 84, 'One free Kaggle T4 each, 0 euro.', 20, cls='s')
    x0, x1, y = 80, W - 80, 170
    b += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{BORDER}" stroke-width="3"/>\n'
    step = (x1 - x0) / (len(pts) - 1)
    for i, (d, name, what) in enumerate(pts):
        x = round(x0 + i * step)
        planned = d == 'next'
        fill = SURFACE if planned else TUNED
        dash = ' stroke-dasharray="4 3"' if planned else ''
        b += f'<circle cx="{x}" cy="{y}" r="11" fill="{fill}" stroke="{TUNED}" stroke-width="3"{dash}/>\n'
        b += t(x, y - 26, d, 18, cls='s', anchor='middle')
        b += t(x, y + 46, name, 20, 700, anchor='middle')
        for j, part in enumerate(what.split(', ')):
            b += t(x, y + 74 + 26 * j, part, 18, cls='s', anchor='middle')
    desc = 'Timeline: ' + '; '.join(f'{d} {n}: {w}' for d, n, w in pts) + '.'
    return svg(y + 136, 'Timeline of the three fine-tuning runs', desc, b)


def terminal():
    lines = (ROOT / 'figures' / 'terminal.txt').read_text(encoding='utf-8').splitlines()
    wrapped = []
    for line in lines:   # wrap at spaces, continuation lines indented; long tokens still break
        wrapped += textwrap.wrap(line, 58, subsequent_indent='  ', break_on_hyphens=False) or ['']
    b = f'<rect x="20" y="20" width="{W - 40}" height="{60 + 28 * len(wrapped)}" rx="10" fill="#0D1117"/>\n'
    for i, colour in enumerate(('#FF5F57', '#FEBC2E', '#28C840')):
        b += f'<circle cx="{48 + 22 * i}" cy="44" r="7" fill="{colour}"/>\n'
    y = 90
    for line in wrapped:
        colour = '#7EE787' if line.startswith('$') else '#E6EDF3'
        b += f'<text x="44" y="{y}" font-size="18" class="m" style="fill: {colour}">{escape(line)}</text>\n'
        y += 28
    desc = 'Terminal transcript: ' + ' / '.join(lines)
    return svg(y + 20, 'Terminal: the extraction skill and the checks, run in this repository', desc, b)


def banner():
    b = t(40, 70, 'lfm3-local-finetuning', 36, 700, 'm')
    b += t(40, 112, 'A 1.7B model on a 6 GB gaming GPU,', 24)
    b += t(40, 144, 'fine-tuned for 0 euro on a free cloud GPU,', 24)
    b += t(40, 176, 'designed to work as the operator of a bigger model.', 24)
    b += f'<rect x="40" y="204" width="120" height="6" rx="3" fill="{TUNED}"/>\n'
    b += t(40, 248, 'Real numbers, small benches, our own test. Measure on yours.', 19, cls='s')
    return svg(280, 'lfm3-local-finetuning', 'A 1.7B model on a 6 GB gaming GPU, fine-tuned for 0 euro on a free cloud GPU, '
               'designed to work as the operator of a bigger model.', b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = claims()
    figures = {'banner.svg': banner(), 'three-numbers.svg': three_numbers(c), 'where-it-loses.svg': where_it_loses(c),
               'orchestrator.svg': orchestrator(), 'hardware.svg': hardware(), 'timeline.svg': timeline(c), 'terminal.svg': terminal()}
    for name, text in figures.items():
        (OUT / name).write_text(text, encoding='utf-8', newline='\n')
        print(name)


if __name__ == '__main__':
    main()
