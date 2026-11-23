# lfm3-local

How to run [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) fast on a normal PC with llama.cpp, and the measurements behind every flag.

On an RTX 4060 that was also running DaVinci Resolve and two Blender windows, the default way of serving it gave 45 to 47 tokens a second. The settings here give 85 to 122, with half the RAM. I picked each flag with `bench.mjs`, alternating the old and new setup in the same session so both saw the same load, and kept only what the numbers backed.

## Run it

```sh
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:QAD-Q4_0 \
  -c 16384 -np 1 -ngl 99 --jinja -lm none -fa auto \
  -bs --samplers "penalties;top_k;temperature" -cram 256
```

Use llama.cpp's CUDA build on NVIDIA cards (b12633 or newer), the Vulkan build on AMD and Intel. Send `temperature 0.1, top_k 50, repeat_penalty 1.1` (Liquid's settings) and `chat_template_kwargs: {"enable_thinking": false}` unless you want it to think first.

## Results

RTX 4060 (8 GB), Ryzen 7 5800X, 32 GB RAM, Windows 11, llama.cpp b12633. Tokens a second while generating, mean of three prompts. "Before" is `-c 16384 -ngl 99 --jinja` on the Vulkan build with the Q4_K_M file, which is how it ran first.

| Setting | GPU busy (normal use) | GPU idle | RAM in use | Video memory (llama.cpp) |
|---|---|---|---|---|
| Before | 45 to 47 tok/s | 65 | 1,443 MB | 1,333 MB |
| These settings | 85 to 122 tok/s (2.1x to 2.7x) | 116 to 126 (1.8x) | 737 MB | 1,171 MB |

- **First answer word:** under 0.1 s with thinking off, against 2 to 20 s when the model thinks first.
- **Ceiling:** `llama-bench` puts the raw limit of this card at 143 tok/s for the Q4_0 file, so on an idle GPU there was not much left to win. LFM3 has no MTP heads and Liquid has no smaller LFM3 to draft with, so speculative decoding cannot push past it.
- **CPU only:** the Q4_0 file generates 2.1x faster than Q4_K_M under the same load.
- **Vulkan:** no faster under load (about 45 tok/s either way), but RAM falls from 1,457 MB to about 400 MB.

## What each flag does here

| Flag | Why | Measured |
|---|---|---|
| CUDA build instead of Vulkan | Vulkan could not run flash attention on this card and slowed badly when other apps shared the GPU | 14 against 68 to 75 tok/s on the same busy GPU |
| `QAD-Q4_0` file | Liquid's quantization-aware 4-bit file; Q4_0 has the fastest kernels | 143 against 131 tok/s raw, 50 MB smaller, 2.1x on CPU |
| `-bs` | samples on the GPU | 116 against 94 tok/s |
| `--samplers penalties;top_k;temperature` | the three samplers Liquid recommends instead of eight | part of the same gain |
| `-np 1` | one conversation slot instead of four | RAM and context cache down |
| `-lm none` | no memory map, so the weights are not also kept in RAM | working RAM 1,459 to 695 MB |
| `-cram 256` | caps the prompt cache, which defaults to 8 GB of RAM | bounded RAM over long chats |
| `-c 16384` | room for a system prompt and tools; LFM3 has only 6 attention layers, so 16k costs 192 MB | |

## Tried and dropped

| Tried | Result |
|---|---|
| Quantized KV cache (`-ctk q8_0 -ctv q8_0`) | 20 to 25% slower to save about 48 MB |
| n-gram speculation (`ngram-cache`, `ngram-mod`) | no gain |
| Smaller batches (`-ub 128`) | 57 MB less video memory, slower on long prompts |
| No pinned memory (`GGML_CUDA_NO_PINNED=1`) | no change |
| High priority (`--prio 2`) | 5% faster, but its busy-waiting threads slow the rest of the PC |

## Reproduce

```sh
node bench.mjs --model=LFM3-1.7B-QAD-Q4_0.gguf \
  --cuda=path/to/cuda/llama-server.exe --vulkan=path/to/vulkan/llama-server.exe \
  --configs=before,final,before,final --quick=1
```

`configs.json` holds every setting above. Run them alternately (`before,final,before,final`): other programs change how busy the GPU is from minute to minute, and a single run of each can mislead. `results.jsonl` has the 91 runs these tables come from.

MIT licensed.
