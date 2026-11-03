"""Synthetic Italian sentences with a date and an amount, and the exact answer ({"data": YYYY-MM-DD|null, "importo_euro": number|null}).

Conventions (the same as bench/extract_bench.jsonl): a date counts only if day, month and year are written; an amount only if it is in
euro (digits with "euro", "EUR", "€", or round amounts in words such as "duemila euro"); percentages, times, days and quantities are not
amounts; with several dates or amounts the first one written wins. Labels are born together with the sentence, so they are exact by
construction. The sentences are Italian because our model works in Italian: swap the templates for your language.
Use: import it (make_sentence(seed) -> (text, answer)); `python data_synthetic/gen_extract.py` prints a sample.
"""
import json
import random
from datetime import date, timedelta

MESI = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
GIORNI = ['lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì', 'sabato', 'domenica']
PAROLE = {100: 'cento', 200: 'duecento', 300: 'trecento', 500: 'cinquecento', 1000: 'mille', 2000: 'duemila', 3000: 'tremila', 5000: 'cinquemila', 10000: 'diecimila'}


def data_a_caso(r):
    return date(2026, 1, 1) + timedelta(days=r.randint(0, 560))


def scrivi_data(r, d):
    s = r.choice(['gg/mm/aaaa', 'g/m/aaaa', 'gg-mm-aaaa', 'gg.mm.aaaa', 'gg/mm/aa', 'iso', 'testo', 'testo', 'testo', 'giorno_testo', 'abbr', 'primo'])
    if s == 'primo' and d.day != 1:
        s = 'testo'
    if s == 'gg/mm/aaaa':
        return f'{d.day:02d}/{d.month:02d}/{d.year}'
    if s == 'g/m/aaaa':
        return f'{d.day}/{d.month}/{d.year}'
    if s == 'gg-mm-aaaa':
        return f'{d.day:02d}-{d.month:02d}-{d.year}'
    if s == 'gg.mm.aaaa':
        return f'{d.day:02d}.{d.month:02d}.{d.year}'
    if s == 'gg/mm/aa':
        return f'{d.day:02d}/{d.month:02d}/{d.year % 100:02d}'
    if s == 'iso':
        return d.isoformat()
    if s == 'giorno_testo':
        return f'{GIORNI[d.weekday()]} {d.day} {MESI[d.month - 1]} {d.year}'
    if s == 'abbr':
        return f'{d.day} {MESI[d.month - 1][:3]}. {d.year}'
    if s == 'primo':
        return f'1° {MESI[d.month - 1]} {d.year}'
    return f'{d.day} {MESI[d.month - 1]} {d.year}'


def importo_a_caso(r):
    t = r.random()
    if t < 0.35:
        return float(r.randint(5, 900))
    if t < 0.65:
        return round(r.uniform(5, 900), 2)
    if t < 0.9:
        return float(r.randint(1, 60) * 100)
    return float(r.choice(list(PAROLE)))


def scrivi_importo(r, x):
    if x in PAROLE.keys() and r.random() < 0.5:
        return f'{PAROLE[int(x)]} euro'
    if x == int(x):
        n = int(x)
        num = f'{n:,}'.replace(',', '.') if n >= 1000 and r.random() < 0.7 else str(n)
        if r.random() < 0.25:
            num += ',00'
    else:
        num = f'{x:,.2f}'.replace(',', '#').replace('.', ',').replace('#', '.') if x >= 1000 else f'{x:.2f}'.replace('.', ',')
    forma = r.choice(['{} euro', '{} euro', '{} €', '{}€', '€ {}', '{} EUR', '{} eur'])
    return forma.format(num)


def valore(x):
    return int(x) if float(x) == int(x) else round(float(x), 2)


DISTRATTORI_DATA = ['domani', 'dopodomani', 'tra 30 giorni', 'entro fine mese', 'il prossimo lunedì', 'il 15 novembre', 'a novembre', 'nel 2027', 'marzo 2026', 'la settimana prossima',
                    'il 3 del mese', 'entro il 20', 'a fine anno', 'il martedì', 'ieri sera']
DISTRATTORI_NUM = ['alle 18:30', 'del 15%', 'con uno sconto del 10%', 'per 3 sessioni', 'in 12 rate', 'al numero 4521', 'tra 5 giorni lavorativi', 'per 2 persone', 'con il 20% di anticipo',
                   'al piano 3', 'alle ore 9', 'in 48 ore', 'in 2 giorni', 'su 7 prodotti']
# {D} = data, {A} = importo; i modelli hanno uno o due segnaposto, e «X» segna un secondo valore (viene dopo: il primo scritto vince)
MODELLI_ENTRAMBI = [
    'La fattura numero {n} di {A} scade il {D}.', 'Il bonifico da {A} è stato fatto il {D}.', 'Appuntamento fissato per {D}, caparra di {A}.', 'Consegna prevista entro il {D}, saldo di {A} alla consegna.',
    'Il preventivo, pari a {A}, vale fino al {D}.', 'Evento il {D} a Bari, biglietto {A}.', 'Pagamento di {A} addebitato il {D}.', 'La rata mensile è di {A}, prima scadenza {D}.',
    'Rimborso di {A} accreditato il {D}.', 'Ci vediamo il {D} per firmare; il compenso concordato è di {A}.', 'Il {D} scade l\'abbonamento da {A} all\'anno.', 'Contratto firmato il {D}: canone di {A} al mese.',
    'Ricevuta del {D}: versati {A} per l\'iscrizione.', 'Il corso inizia il {D} e costa {A}.', 'Penale di {A} se si disdice dopo il {D}.', 'Saldo di {A} da versare entro il {D}.',
    'Offerta valida fino al {D}: {A} in meno sul totale.', 'In data {D} è stato emesso un buono da {A}.', 'Il noleggio dal {D} costa {A} al giorno.', 'Stipendio di {A} accreditato il {D}.',
    'La multa da {A} va pagata entro il {D}.', 'Hanno anticipato {A} il {D} per bloccare la prenotazione.', 'Verbale del {D}: approvata una spesa di {A} per la nuova attrezzatura.',
    'Spesa di {A} registrata nel rapporto del {D}.', 'Il fornitore ha confermato {A} per la fornitura, con consegna il {D}.', 'Nota di credito del {D} per {A}.']
MODELLI_DATA = ['Riunione spostata a {D}, nessun costo.', 'Appuntamento il {D} alle 15.', 'La scadenza è il {D}.', 'Ricordo che il termine è il {D}.', 'La consegna è avvenuta il {D} senza problemi.',
                'Evento in programma il {D}, ingresso libero.', 'Il {D} è stato approvato il nuovo calendario.', 'Il sopralluogo è fissato per {D}, nessun importo da versare.',
                'La riunione del {D} è stata rinviata.', 'Scadenza del contratto: {D}.', 'Verbale redatto il {D}, firmato da tutti i presenti.']
MODELLI_IMPORTO = ['Il canone di {A} al mese non è ancora stato aggiornato.', 'Il pacchetto costa {A}.', 'Quota di iscrizione: {A}.', 'Il totale è {A} iva inclusa.', 'Servono circa {A} per i materiali.',
                   'L\'abbonamento costa {A} all\'anno.', 'Costo del trasporto: {A}.', 'Il listino prevede {A} per ogni intervento.', 'Il budget approvato è di {A}.', 'Spedizione a carico del cliente: {A}.']
MODELLI_NULLA = ['Spedizione entro 5 giorni lavorativi, nessun costo aggiuntivo.', 'Il fatturato è cresciuto del 12% nel 2026.', 'La riunione è confermata, ci vediamo in sede.',
                 'Il cliente ha chiesto una revisione del testo.', 'Il preventivo arriva domani, senza impegno.', 'Il lavoro è stato consegnato in anticipo.', 'Abbiamo ricevuto 3 richieste di informazioni.',
                 'Lo sconto è del 15% su tutto il listino.', 'Il corso dura 12 ore in 4 incontri.', 'Il volo parte alle 7:45 dal terminal 2.', 'La consegna slitta a data da definire.',
                 'Nessuna scadenza per questa pratica.', 'Il prezzo sarà comunicato a voce.', 'Chiamare il fornitore per sapere i tempi.']


def make_sentence(seme):
    """(testo, {"data": ..., "importo_euro": ...})."""
    r = random.Random(seme)
    k = r.random()
    d = data_a_caso(r)
    x = importo_a_caso(r)
    n = r.randint(10, 999)
    if k < 0.45:
        t = r.choice(MODELLI_ENTRAMBI).format(n=n, D=scrivi_data(r, d), A=scrivi_importo(r, x))
        ris = (d.isoformat(), valore(x))
    elif k < 0.60:
        t = r.choice(MODELLI_DATA).format(D=scrivi_data(r, d))
        ris = (d.isoformat(), None)
    elif k < 0.75:
        t = r.choice(MODELLI_IMPORTO).format(A=scrivi_importo(r, x))
        ris = (None, valore(x))
    elif k < 0.85:
        t = r.choice(MODELLI_NULLA)
        ris = (None, None)
    elif k < 0.93:
        # una data senza anno o relativa e un importo: la data resta null
        t = r.choice(['{R}, {A} da versare.', 'Il saldo di {A} va versato {R}.', '{R} si paga la rata di {A}.']).format(R=r.choice(DISTRATTORI_DATA), A=scrivi_importo(r, x))
        t = t[0].upper() + t[1:]
        ris = (None, valore(x))
    else:
        # due date o due importi: vale il primo scritto
        d2 = d + timedelta(days=r.randint(7, 90))
        x2 = importo_a_caso(r)
        if r.random() < 0.5:
            t = r.choice(['Pagato in due rate da {A}, il {D} e il {E}.', 'Dal {D} al {E} il canone è di {A}.', 'Prima scadenza il {D}, seconda scadenza il {E}: rate di {A}.']).format(
                A=scrivi_importo(r, x), D=scrivi_data(r, d), E=scrivi_data(r, d2))
            ris = (d.isoformat(), valore(x))
        else:
            t = r.choice(['Il listino passa da {A} a {B} dal {D}.', 'Il totale è {A} per i pezzi, più {B} di spedizione, ritiro il {D}.', 'Costo iniziale {A}, poi {B} al mese, partenza il {D}.']).format(
                A=scrivi_importo(r, x), B=scrivi_importo(r, x2), D=scrivi_data(r, d))
            ris = (d.isoformat(), valore(x))
    if r.random() < 0.3 and ris[1] is None:
        t = t[:-1] + ' ' + r.choice(DISTRATTORI_NUM) + '.' if t.endswith('.') else t
    return t, {'data': ris[0], 'importo_euro': ris[1]}


if __name__ == '__main__':
    for s in range(24):
        t, ris = make_sentence(s)
        print(t, '->', json.dumps(ris, ensure_ascii=False))
