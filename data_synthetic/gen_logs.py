"""Synthetic logs with an exact per-line label ("rilevante" = relevant / "rumore" = noise), to train and test a log filter.

Labelling rule (the same one the model prompt states): a line is RELEVANT if it is an error, a failure, an exception or its message,
the name of a failing test, the file position that caused the problem, a final summary (counts, outcome, total time, exit code)
or the state of a service (listening / does not start). Everything else is NOISE: progress, passed tests, headers, versions,
configuration, code context lines, deprecation warnings, repetitions.
Labels are born together with the line (no judgement afterwards), so they are exact by construction.
Nine log families: pytest, vitest, ffmpeg, git, pip/npm install, Next.js build, llama.cpp server, PowerShell, application log.
Labels and some messages are in Italian because our model works in Italian; the log text itself is mostly English, as real logs are.
Use: import it (make_document(seed) -> (family, [(line, label)])); `python data_synthetic/gen_logs.py` prints one example per family.
Paths, repositories and branch names are invented.
"""
import random

R, N = 'rilevante', 'rumore'
MODULI = ['client', 'parser', 'estrai', 'regole', 'orari', 'prezzi', 'calendario', 'utenti', 'fatture', 'ordini', 'api', 'scheduler', 'mailer', 'cache',
          'sincronia', 'rapporti', 'magazzino', 'lead', 'campagne', 'landing', 'verifica', 'registro', 'permessi', 'esportazione']
VERBI = ['legge', 'scrive', 'valida', 'calcola', 'filtra', 'ordina', 'unisce', 'divide', 'converte', 'cancella', 'aggiorna', 'rifiuta', 'accetta', 'ricarica']
OGGETTI = ['data', 'importo', 'cliente', 'riga', 'token', 'lotto', 'codice', 'percorso', 'scadenza', 'brand', 'nota', 'ordine', 'foglio', 'modulo']
CONDIZIONI = ['vuoto', 'con_accenti', 'doppio', 'senza_anno', 'troppo_lungo', 'negativo', 'nullo', 'in_ritardo', 'duplicato', 'dopo_mezzanotte']
ECCEZIONI = ['ValueError: invalid literal for int() with base 10: ', 'KeyError: ', 'AssertionError: ', 'TypeError: unsupported operand type(s) for +: ',
             'FileNotFoundError: [Errno 2] No such file or directory: ', 'json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)',
             'IndexError: list index out of range', 'AttributeError: NoneType object has no attribute ', 'ZeroDivisionError: division by zero',
             'UnicodeDecodeError: charmap codec cannot decode byte 0x9d in position ', 'PermissionError: [Errno 13] Permission denied: ']
CARTELLE = ['/srv/app', '/opt/project', '/work/api', '/work/web', '/data/render']  # invented paths


def _ora(r):
    return f'{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}'


def _nome_test(r):
    return f'test_{r.choice(VERBI)}_{r.choice(OGGETTI)}_{r.choice(CONDIZIONI)}'


def _eccezione(r):
    e = r.choice(ECCEZIONI)
    if e.endswith(': ') or e.endswith('position ') or e.endswith('attribute '):
        e += r.choice(["'x'", "'2026-13-45'", "'abc'", "'brand'", "'importo'", "'dati/lista.csv'", '12', "'vuoto'", '3'])
    return e


def pytest_log(r):
    righe = []
    ne = r.randint(0, 3)
    righe += [('=' * 29 + ' test session starts ' + '=' * 29, N), (f'platform win32 -- Python 3.{r.randint(11, 14)}.{r.randint(0, 9)}, pytest-8.{r.randint(0, 4)}.{r.randint(0, 3)}, pluggy-1.{r.randint(3, 6)}.0', N),
              (f'rootdir: {r.choice(CARTELLE)}', N), (f'plugins: anyio-4.{r.randint(0, 9)}.0, cov-5.0.0', N), (f'collected {r.randint(20, 400)} items', N)]
    falliti = []
    tot = 0
    for _ in range(r.randint(4, 12)):
        f = f'test/test_{r.choice(MODULI)}.py'
        n = r.randint(3, 30)
        tot += n
        esiti = ['.'] * n
        guasti = r.sample(range(n), min(ne, n)) if (ne and r.random() < 0.4) else []
        for g in guasti:
            esiti[g] = r.choice('FE')
            falliti.append((f, _nome_test(r)))
        righe.append((f'{f} {"".join(esiti)}'.ljust(40) + f'[{r.randint(1, 100):3d}%]', R if guasti else N))
        ne = max(0, ne - len(guasti))
    if falliti:
        righe.append(('=' * 35 + ' FAILURES ' + '=' * 35, N))
        for f, t in falliti:
            righe.append(('_' * 16 + f' {t} ' + '_' * 16, R))
            righe += [(f'    def {t}():', N), (f'        r = {r.choice(MODULI)}({r.randint(0, 9)})', N)]
            esc = r.random()
            if esc < 0.5:
                righe += [(f'>       assert r == {r.randint(0, 9)}', R), (f'E       assert {r.randint(0, 9)} == {r.randint(0, 9)}', R)]
                finale = 'AssertionError'
            else:
                righe += [(f'>       r = {r.choice(MODULI)}.{r.choice(VERBI)}({r.randint(0, 9)})', R), (f'E       {_eccezione(r)}', R)]
                finale = 'Error'
            righe.append((f'{f}:{r.randint(10, 300)}: {finale}', R))
        righe.append(('=' * 18 + ' short test summary info ' + '=' * 18, N))
        righe += [(f'FAILED {f}::{t} - {r.choice(["assert", "AssertionError", "ValueError", "KeyError"])}', R) for f, t in falliti]
        righe.append(('=' * 20 + f' {len(falliti)} failed, {tot - len(falliti)} passed in {r.uniform(0.5, 90):.2f}s ' + '=' * 20, R))
    else:
        if r.random() < 0.4:
            righe += [('=' * 14 + ' warnings summary ' + '=' * 14, N), (f'test/test_{r.choice(MODULI)}.py:{r.randint(5, 90)}: DeprecationWarning: datetime.utcnow() is deprecated', N),
                      ('-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html', N)]
        righe.append(('=' * 20 + f' {tot} passed in {r.uniform(0.5, 90):.2f}s ' + '=' * 20, R))
    return righe


def vitest_log(r):
    righe = [(f' RUN  v{r.randint(1, 3)}.{r.randint(0, 6)}.{r.randint(0, 9)} {r.choice(CARTELLE)}', N)]
    guasti = []
    for _ in range(r.randint(5, 14)):
        f = f'src/{r.choice(MODULI)}/{r.choice(MODULI)}.test.ts'
        n = r.randint(2, 14)
        if r.random() < 0.18:
            guasti.append((f, f'{r.choice(OGGETTI)} {r.choice(VERBI)}'))
            righe.append((f' ❯ {f} ({n} tests | 1 failed) {r.randint(5, 400)}ms', R))
            righe.append((f'   × {guasti[-1][1]} {r.randint(1, 30)}ms', R))
        else:
            righe.append((f' ✓ {f} ({n} tests) {r.randint(2, 400)}ms', N))
    for f, nome in guasti:
        righe += [(f' FAIL  {f} > {nome}', R), (f'AssertionError: expected {r.randint(0, 9)} to be {r.randint(0, 9)} // Object.is equality', R),
                  (f' ❯ {f}:{r.randint(5, 90)}:{r.randint(3, 40)}', R), (f'     {r.randint(5, 30)}|   const r = {r.choice(MODULI)}({r.randint(0, 5)})', N),
                  (f'     {r.randint(5, 30)}|   expect(r).toBe({r.randint(0, 9)})', N)]
    tf = r.randint(8, 40)
    righe += [(f' Test Files  {len(guasti)} failed | {tf} passed ({tf + len(guasti)})' if guasti else f' Test Files  {tf} passed ({tf})', R),
              (f'      Tests  {len(guasti)} failed | {r.randint(40, 400)} passed' if guasti else f'      Tests  {r.randint(40, 400)} passed', R),
              (f'   Start at  {_ora(r)}', N), (f'   Duration  {r.uniform(1, 40):.2f}s (transform {r.randint(50, 900)}ms, collect {r.randint(100, 3000)}ms)', R)]
    return righe


def ffmpeg_log(r):
    righe = [(f'ffmpeg version {r.randint(5, 8)}.{r.randint(0, 1)} Copyright (c) 2000-2025 the FFmpeg developers', N), ('  built with gcc 14.2.0 (Rev1, Built by MSYS2 project)', N),
             ('  configuration: --enable-gpl --enable-libx264 --enable-libx265 --enable-libvpx --enable-nvenc', N),
             ('  libavutil      60.  8.100 / 60.  8.100', N), ('  libavcodec     62. 11.100 / 62. 11.100', N)]
    errore = r.random() < 0.4
    nome = f'{r.choice(["clip", "scena", "reel", "voce", "fondo"])}_{r.randint(1, 40)}.{r.choice(["mp4", "mov", "wav"])}'
    righe += [(f"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '{nome}':", N), ('  Duration: 00:00:%02d.%02d, start: 0.000000, bitrate: %d kb/s' % (r.randint(2, 59), r.randint(0, 99), r.randint(300, 9000)), N),
              (f'  Stream #0:0[0x1](und): Video: h264 (High), yuv420p, 1080x1920, {r.randint(2000, 9000)} kb/s, 30 fps', N),
              ('  Stream #0:1[0x2](und): Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s', N), (f'Output #0, mp4, to \'{r.choice(["out", "finale", "render"])}.mp4\':', N)]
    if errore:
        tipo = r.choice(['file', 'decode', 'filtro', 'codec'])
        if tipo == 'file':
            righe += [(f'{nome}: No such file or directory', R)]
        elif tipo == 'decode':
            righe += [(f'[h264 @ 0x{r.randint(0x1000, 0xffff):x}] error while decoding MB {r.randint(1, 99)} {r.randint(1, 99)}, bytestream -{r.randint(1, 9)}', R),
                      ('Error while decoding stream #0:0: Invalid data found when processing input', R)]
        elif tipo == 'filtro':
            righe += [("[AVFilterGraph @ 0x%x] No such filter: 'subtitle'" % r.randint(0x1000, 0xffff), R), ('Error initializing filters!', R)]
        else:
            righe += [("Unknown encoder 'h264_nvenc'", R), ('[vost#0:0/h264_nvenc @ 0x%x] Error selecting an encoder' % r.randint(0x1000, 0xffff), R)]
        righe += [('Conversion failed!', R)]
    else:
        for k in range(r.randint(4, 12)):
            righe.append((f'frame={r.randint(10, 3000):5d} fps={r.randint(20, 200):3d} q=28.0 size={r.randint(100, 20000):7d}kB time=00:00:{k:02d}.00 bitrate={r.randint(500, 9000)}.0kbits/s speed={r.uniform(0.5, 6):.2f}x', N))
        righe += [(f'[libx264 @ 0x{r.randint(0x1000, 0xffff):x}] frame I:{r.randint(1, 30)}   Avg QP:{r.uniform(10, 30):.2f}  size: {r.randint(1000, 90000)}', N),
                  (f'video:{r.randint(100, 30000)}kB audio:{r.randint(10, 900)}kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: {r.uniform(0.1, 2):.6f}%', R)]
    return righe


def git_log(r):
    ramo = r.choice(['main', 'feature/login', 'fix/parser', 'dev', 'release/2.1'])
    esito = r.choice(['push_ok', 'push_rifiutato', 'merge_ok', 'merge_conflitto', 'rebase_conflitto'])
    righe = []
    if esito.startswith('push'):
        righe += [('Enumerating objects: %d, done.' % r.randint(5, 400), N), ('Counting objects: 100%% (%d/%d), done.' % ((n := r.randint(5, 400)), n), N),
                  ('Delta compression using up to 12 threads', N), ('Compressing objects: 100%% (%d/%d), done.' % ((m := r.randint(3, 200)), m), N),
                  ('Writing objects: 100%% (%d/%d), %.2f KiB | 1.20 MiB/s, done.' % ((k := r.randint(3, 200)), k, r.uniform(1, 900)), N), ('Total %d (delta %d), reused 0 (delta 0), pack-reused 0' % (k, r.randint(0, 90)), N)]
        if esito == 'push_ok':
            righe += [(f'To https://github.com/example-org/{r.choice(["app", "api", "web", "tools"])}.git', N), (f'   {r.randint(0x1000000, 0xfffffff):x}..{r.randint(0x1000000, 0xfffffff):x}  {ramo} -> {ramo}', R)]
        else:
            righe += [(f'To https://github.com/example-org/{r.choice(["app", "api", "web", "tools"])}.git', N), (f' ! [rejected]        {ramo} -> {ramo} (fetch first)', R),
                      (f"error: failed to push some refs to 'https://github.com/example-org/{r.choice(['app', 'api', 'web'])}.git'", R),
                      ('hint: Updates were rejected because the remote contains work that you do not', R), ('hint: have locally. Integrate the remote changes before pushing again.', R),
                      ("hint: See the 'Note about fast-forwards' in 'git push --help' for details.", N)]
    else:
        righe += [(f'From https://github.com/example-org/{r.choice(["app", "api", "web"])}', N), (f' * branch            {ramo}       -> FETCH_HEAD', N)]
        for _ in range(r.randint(2, 6)):
            righe.append((f'Auto-merging {r.choice(["src", "test", "strumenti", "docs"])}/{r.choice(MODULI)}.{r.choice(["py", "ts", "md"])}', N))
        if esito == 'merge_ok':
            righe += [("Merge made by the 'ort' strategy.", N), (f' {r.randint(2, 30)} files changed, {r.randint(10, 900)} insertions(+), {r.randint(0, 200)} deletions(-)', R)]
        else:
            for _ in range(r.randint(1, 3)):
                righe.append((f'CONFLICT (content): Merge conflict in {r.choice(["src", "test", "strumenti"])}/{r.choice(MODULI)}.{r.choice(["py", "ts"])}', R))
            righe.append(('Automatic merge failed; fix conflicts and then commit the result.' if esito == 'merge_conflitto' else 'error: could not apply %x... messaggio del commit' % r.randint(0x1000000, 0xfffffff), R))
    return righe


def installa_log(r):
    npm = r.random() < 0.5
    righe = []
    ok = r.random() < 0.6
    if npm:
        for _ in range(r.randint(3, 8)):
            righe.append((f'npm warn deprecated {r.choice(["inflight", "glob", "rimraf", "uuid", "request"])}@{r.randint(1, 9)}.{r.randint(0, 9)}.{r.randint(0, 9)}: This package is no longer supported.', N))
        if ok:
            righe += [(f'added {r.randint(20, 900)} packages, and audited {r.randint(20, 900)} packages in {r.randint(2, 60)}s', R), (f'{r.randint(5, 200)} packages are looking for funding', N), ('  run `npm fund` for details', N)]
            v = r.randint(0, 6)
            righe.append(('found 0 vulnerabilities' if v == 0 else f'{v} vulnerabilities ({v // 2} moderate, {v - v // 2} high)', R))
        else:
            righe += [('npm error code ERESOLVE', R), ('npm error ERESOLVE could not resolve', R), (f'npm error While resolving: {r.choice(MODULI)}@{r.randint(1, 5)}.{r.randint(0, 9)}.0', R),
                      ('npm error Fix the upstream dependency conflict, or retry', N), ('npm error this command with --force or --legacy-peer-deps', N),
                      (f'npm error A complete log of this run can be found in: ~/.npm/_logs/2026-10-{r.randint(1, 30):02d}T{r.randint(0, 23):02d}_debug-0.log', N)]
    else:
        pacchetti = [r.choice(['requests', 'numpy', 'pandas', 'peft', 'transformers', 'accelerate', 'pyyaml', 'pillow', 'fastapi']) + f'-{r.randint(0, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}' for _ in range(r.randint(3, 7))]
        for p in pacchetti:
            righe += [(f'Collecting {p.split("-")[0]}>={p.split("-")[1]}', N), (f'  Downloading {p}-py3-none-any.whl ({r.randint(10, 9000)} kB)', N),
                      ('     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ %d.%d/%d.%d kB %d.%d MB/s eta 0:00:00' % (r.randint(1, 9), r.randint(0, 9), r.randint(1, 9), r.randint(0, 9), r.randint(1, 9), r.randint(0, 9)), N)]
        if r.random() < 0.3:
            righe.append(('  Using cached ' + pacchetti[0] + '-py3-none-any.whl', N))
        if ok:
            righe += [('Installing collected packages: ' + ', '.join(p.split('-')[0] for p in pacchetti), N), ('Successfully installed ' + ' '.join(pacchetti), R)]
        else:
            righe += [(f'ERROR: Could not find a version that satisfies the requirement {r.choice(MODULI)}==9.{r.randint(0, 9)}.0 (from versions: none)', R),
                      (f'ERROR: No matching distribution found for {r.choice(MODULI)}==9.{r.randint(0, 9)}.0', R)]
        if r.random() < 0.5:
            righe.append((f'[notice] A new release of pip is available: 25.{r.randint(0, 3)} -> 26.{r.randint(0, 2)}', N))
    return righe


def build_log(r):
    righe = [(f'   ▲ Next.js {r.randint(14, 16)}.{r.randint(0, 3)}.{r.randint(0, 9)}', N), ('   Creating an optimized production build ...', N)]
    if r.random() < 0.6:
        righe += [(f' ✓ Compiled successfully in {r.uniform(3, 60):.1f}s', R), ('   Linting and checking validity of types ...', N), (' ✓ Collecting page data', N), (' ✓ Generating static pages (%d/%d)' % ((n := r.randint(5, 80)), n), N),
                  ('Route (app)                              Size     First Load JS', N)]
        for _ in range(r.randint(4, 10)):
            righe.append((f'├ ○ /{r.choice(MODULI)}{"/" + r.choice(OGGETTI) if r.random() < 0.4 else ""}'.ljust(40) + f'{r.uniform(0.2, 40):.1f} kB   {r.randint(90, 300)} kB', N))
        righe += [('○  (Static)  prerendered as static content', N), ('ƒ  (Dynamic)  server-rendered on demand', N)]
    else:
        f = f'./src/app/{r.choice(MODULI)}/page.tsx'
        righe += [(' ⨯ Failed to compile.', R), ('', N), (f'{f}:{r.randint(3, 120)}:{r.randint(2, 40)}', R),
                  (r.choice([f"Type error: Property '{r.choice(OGGETTI)}' does not exist on type '{r.choice(['Cliente', 'Ordine', 'Lead'])}'.", f"Module not found: Can't resolve '@/lib/{r.choice(MODULI)}'",
                             f"Type error: Argument of type 'string' is not assignable to parameter of type 'number'."]), R),
                  (f'  {r.randint(3, 120) - 1} | const x = {r.choice(MODULI)}({r.randint(0, 3)})', N), (f'> {r.randint(3, 120)} | return <{r.choice(["Tabella", "Scheda", "Elenco"])} dati={{x}} />', N), ('      |         ^', N),
                  ('Next.js build worker exited with code: 1 and signal: null', R)]
    return righe


def server_log(r):
    righe = []
    for k in range(r.randint(6, 16)):
        righe.append((f'llama_model_loader: - kv {k:3d}: {r.choice(["general.name", "lfm3.block_count", "lfm3.context_length", "tokenizer.ggml.model", "general.file_type"])} {r.choice(["str", "u32", "bool"])} = {r.choice(["lfm3", "32", "128000", "gpt2", "7"])}', N))
    righe += [(f'load_tensors: offloaded {n}/{n} layers to GPU' if (n := r.randint(20, 40)) else '', N), (f'llama_context: n_ctx = {r.choice([8192, 16384, 32768])}', N)]
    sorte = r.choice(['ok', 'ok', 'oom', 'file', 'porta'])
    if sorte == 'ok':
        righe += [(f'main: server is listening on http://127.0.0.1:{r.choice([8080, 8081])}', R)]
        for _ in range(r.randint(5, 25)):
            righe.append((f'srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200', N))
            righe.append((f'slot print_timing: id  {r.randint(0, 1)} | task {r.randint(0, 9000)} | prompt eval time = {r.uniform(20, 900):.2f} ms / {r.randint(30, 900)} tokens', N))
    elif sorte == 'oom':
        righe += [('ggml_backend_cuda_buffer_type_alloc_buffer: allocating %.2f MiB on device 0: cudaMalloc failed: out of memory' % r.uniform(500, 4000), R), ('llama_model_load: error loading model: unable to allocate CUDA0 buffer', R),
                  ('main: exiting due to model loading error', R)]
    elif sorte == 'file':
        righe += [('gguf_init_from_file: failed to open GGUF file \'lfm3-x.gguf\'', R), ('llama_model_load: error loading model: llama_model_loader: failed to load model from lfm3-x.gguf', R), ('main: exiting due to model loading error', R)]
    else:
        righe += [(f'couldn\'t bind HTTP server socket, hostname: 127.0.0.1, port: {r.choice([8080, 8081])}', R), ('main: exiting due to HTTP server error', R)]
    return [x for x in righe if x[0]]


def powershell_log(r):
    percorso = r.choice(CARTELLE) + '/' + r.choice(MODULI)
    righe = []
    for _ in range(r.randint(1, 3)):
        c = r.choice(['Get-Item', 'Copy-Item', 'Remove-Item', 'Get-Content'])
        righe += [(f"{c} : Cannot find path '{percorso}/{r.choice(OGGETTI)}.txt' because it does not exist.", R), (f'At {r.choice(CARTELLE)}/{r.choice(MODULI)}.ps1:{r.randint(1, 80)} char:{r.randint(1, 30)}', R),
                  (f'+ {c} {percorso}', N), ('+ ~~~~~~~~~~~~~~~~~~~~~~~~', N), ('    + CategoryInfo          : ObjectNotFound: (%s:String) [%s], ItemNotFoundException' % (percorso, c), N),
                  ('    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.%sCommand' % c.replace('-', ''), N)]
    if r.random() < 0.5:
        righe.append((f'Exit code: {r.randint(1, 3)}', R))
    return righe


def app_log(r):
    righe = []
    guasto = r.random() < 0.55
    pos = r.randint(8, 30) if guasto else -1
    for k in range(r.randint(18, 45)):
        t = f'2026-10-{r.randint(1, 30):02d} {_ora(r)}'
        if k == pos:
            tipo = r.choice(['traceback', 'error', 'warn'])
            if tipo == 'traceback':
                righe += [(f'{t} ERROR  worker-{r.randint(1, 6)} job {r.randint(1000, 9999)} failed', R), ('Traceback (most recent call last):', R)]
                for _ in range(r.randint(1, 3)):
                    righe += [(f'  File "{r.choice(CARTELLE)}/{r.choice(MODULI)}.py", line {r.randint(5, 300)}, in {r.choice(VERBI)}_{r.choice(OGGETTI)}', N), (f'    {r.choice(OGGETTI)} = {r.choice(MODULI)}.{r.choice(VERBI)}({r.choice(OGGETTI)})', N)]
                righe += [(f'  File "{r.choice(CARTELLE)}/{r.choice(MODULI)}.py", line {r.randint(5, 300)}, in {r.choice(VERBI)}_{r.choice(OGGETTI)}', R), (f'    {r.choice(OGGETTI)} = int({r.choice(OGGETTI)})', R), (_eccezione(r), R)]
            elif tipo == 'error':
                righe.append((f'{t} ERROR  {r.choice(MODULI)}: {r.choice(["connection refused", "timeout dopo 30s", "risposta 500 dal servizio", "chiave mancante", "disco pieno", "migrazione non applicata"])} ({r.choice(OGGETTI)} {r.randint(1, 900)})', R))
            else:
                righe.append((f'{t} WARN   {r.choice(MODULI)}: {r.choice(["riprovo 3/5 dopo errore di rete", "quota quasi esaurita (92%)", "token in scadenza tra 2 giorni", "file non trovato, uso il valore predefinito"])}', R))
        else:
            livello = r.choice(['INFO ', 'INFO ', 'INFO ', 'DEBUG'])
            msg = r.choice([f'avvio worker {r.randint(1, 8)}', f'richiesta ok 200 in {r.randint(2, 300)}ms', f'cache hit {r.choice(OGGETTI)}', f'letti {r.randint(1, 900)} record da {r.choice(MODULI)}',
                            f'job {r.randint(1000, 9999)} completato', 'heartbeat', f'connessione aperta ({r.randint(1, 20)} attive)', f'scritto {r.choice(MODULI)}.json'])
            righe.append((f'{t} {livello}  {r.choice(MODULI)}: {msg}', N))
    if guasto:
        righe.append((f'{_ora(r)} INFO  shutdown con {r.randint(1, 9)} job falliti', R))
    return righe


FAMIGLIE = [pytest_log, vitest_log, ffmpeg_log, git_log, installa_log, build_log, server_log, powershell_log, app_log]


def make_document(seme, famiglie=None):
    """Un documento di log: (famiglia, [(riga, etichetta), ...])."""
    r = random.Random(seme)
    f = (famiglie or FAMIGLIE)[r.randrange(len(famiglie or FAMIGLIE))]
    return f.__name__, f(r)


if __name__ == '__main__':
    for i, f in enumerate(FAMIGLIE):
        nome, righe = make_document(i, [f])
        print('=====', nome, len(righe), 'righe,', sum(e == R for _, e in righe), 'rilevanti')
        for riga, e in righe[:14]:
            print(('R ' if e == R else '. ') + riga[:110])
