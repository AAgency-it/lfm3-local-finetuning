# From the merged model to a local server

The training scripts end with a **merged fp16 model** (config, safetensors weights, tokenizer) in the folder given by `--out`.
To use it with llama.cpp you convert it to GGUF, quantise it and serve it in place of the base model.

> These are llama.cpp's standard tools. We did not record our exact command lines when we converted our two models, so the lines
> below are the usual ones, not a transcript, and **we have not re-run them as written on b12002**. Check them against your version.

## Names
- Weights used for training: Hugging Face `LiquidAI/LFM3-1.7B` (the scripts download it; log in to Hugging Face first if the page asks you to).
- Base model we serve: Hugging Face `LiquidAI/LFM3-1.7B-GGUF`, file `LFM3-1.7B-Q4_K_M.gguf`.
- Both under the LFM Open License v1.0: see [NOTICE](../NOTICE).

## 1. Convert
Use the Python converter from the llama.cpp **source** at the same version as your `llama-server` binary (we used b12002: an older
build did not know the `lfm3` architecture). The converter has its own Python requirements:
```
git clone --depth 1 --branch b12002 https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py <merged model folder> --outfile lfm3-ft-f16.gguf --outtype f16
```
(If the requirements file has another name in your version, look in `llama.cpp/requirements/`.)

## 2. Quantise
Our fine-tuned models are Q8_0 (1,846,237,728 bytes each, about 1.8 GB); the base is Q4_K_M. Remember this when you compare them.
```
llama-quantize lfm3-ft-f16.gguf lfm3-ft-q8_0.gguf Q8_0
```

## 3. Serve it in place of the base
Base on port 8018, fine-tuned on port 8020. On a 6 GB card we ran **one server at a time**: stop the base first.
```
llama-server -m lfm3-ft-q8_0.gguf -c 8192 -np 1 -ngl 99 --host 127.0.0.1 --port 8020 --cache-ram 0 --chat-template-kwargs '{"enable_thinking": false}'
curl http://127.0.0.1:8020/health
```
Windows PowerShell 5.1 needs the JSON quoted differently: `--chat-template-kwargs '{\"enable_thinking\": false}'`.

## 4. Point the skills at it and measure
```
export LFM_SERVER=http://127.0.0.1:8020      # PowerShell: $env:LFM_SERVER = "http://127.0.0.1:8020"
export PYTHONPATH=src
python bench/measure_extract.py
```
Measure the base and the fine-tuned model with the same bench, the same script and the same no-model baseline, and keep both result files.
