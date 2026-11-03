# Install and run LFM3 1.7B locally

Goal: a local, OpenAI-compatible server on `127.0.0.1:8018`, with thinking off, using about 1.4 GB of GPU memory (1,401 MiB measured
on 2026-11-03, see `results.md`).
Tested by us on Windows 11 with an NVIDIA GPU (6 GB) and the CUDA 12.4 build of llama.cpp. Linux and macOS use the same flags
with a different binary; we have not run them.

## 1. Get llama.cpp
Download a **recent** release of llama.cpp for your platform from https://github.com/ggml-org/llama.cpp/releases. We used **b12002**
(`llama-b12002-...-cuda124`); an older build (b11081) did not know the `lfm3` architecture. If your build is older than the one
that added `lfm3`, update it before anything else.

## 2. Get the model
Download `LFM3-1.7B-Q4_K_M.gguf` from the Hugging Face repository `LiquidAI/LFM3-1.7B-GGUF`.
The model is Liquid AI's, under the *LFM Open License v1.0*
(see `NOTICE`; read the conditions that apply to you before using it commercially or redistributing weights).

## 3. Start the server
```
llama-server -m LFM3-1.7B-Q4_K_M.gguf -c 16384 -np 2 -ngl 99 --host 127.0.0.1 --port 8018 --cache-ram 0 --chat-template-kwargs '{"enable_thinking": false}'
```
- `-c 16384 -np 2`: 16k context split in two slots (8k each). Use one slot if you only send one request at a time.
- `-ngl 99`: all layers on the GPU. On a CPU-only machine drop it and expect it to be slower; we did not measure CPU-only.
- `--host 127.0.0.1`: do not expose it to the network; there is no authentication.
- `--cache-ram 0`: no prompt cache in system RAM (see pitfall 12 in `pitfalls.md`).
- Windows PowerShell 5.1 needs the JSON quoted differently: `--chat-template-kwargs '{\"enable_thinking\": false}'`.

Check:
```
curl http://127.0.0.1:8018/health
```
should answer `{"status":"ok"}` once the model is loaded.

## 4. Use it
```
# nothing to install: the library is standard-library only. From the repository root:
export PYTHONPATH=src     # PowerShell: $env:PYTHONPATH = "src"
python -c "from lfm3_local import extract; print(extract.run('Il bonifico da 480 euro è del 3/12/2026.'))"
```
Environment variables: `LFM_SERVER` (default `http://127.0.0.1:8018`), `LFM_LOG` (folder for the usage log, default `./lfm-log`;
the log keeps skill, fingerprint of the text, outcome and time, **never the text**).

## 5. Run the tests and the bench
```
python -m unittest discover -s tests          # no server needed
python bench/measure_extract.py                # server needed
```

## What it costs
Zero in tokens or API fees, the text never leaves your machine. The price is quality: read `results.md` before you give it any job.
