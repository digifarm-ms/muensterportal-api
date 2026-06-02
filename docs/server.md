# muenster4you.de Server

Hardware specs and notes on which models can actually run on this server.
All measurements are CPU-only — there is no GPU.

## Hardware

| Component | Value |
|-----------|-------|
| OS        | Debian GNU/Linux |
| Kernel    | `6.1.0-43-amd64` (Debian 6.1.162-1, 2026-02-08) |
| Arch      | `x86_64` |
| Hostname  | `v2202502255269316888` |
| CPU       | AMD EPYC-Rome Processor |
| CPU cores | 4 (online 0–3, 1 thread/core, single NUMA node) |
| GPU       | none |
| RAM       | 7.8 GiB total |
| Free RAM  | ~5.4 GiB at idle |
| Swap      | 0 B (no swap configured) |
| Disk      | `/dev/vda3` — 251 GB total, ~161 GB free (~34 % used) mounted at `/` |
| `uv`      | 0.7.6 |

Raw `free -h` at idle:

```
              gesamt  benutzt  frei    gemns.  Puffer/Cache  verfügbar
Speicher:     7,8Gi   2,3Gi    243Mi   153Mi   5,6Gi         5,4Gi
Swap:         0B      0B       0B
```

## Embedding models — what we tried

All runs were CPU-only with `sentence-transformers` / HF loaders, no quantization.

### Qwen3-Embedding-4B — OOM

Killed at 54 % of weight loading, no Python traceback (silent kernel OOM
kill). With ~5.3 GiB free and fp16 weights, 4B params need ~8 GiB just for
parameters — does not fit.

```
[pre]   RSS=0.76 GB, avail=5.32 GB
Loading weights:  54%|█████▍    | 215/398 [00:03<00:02, 65.07it/s]
<killed>
```

### Qwen3-Embedding-1.7B — does not exist

The Qwen3-Embedding family only ships 0.6B / 4B / 8B variants. 1.7B is not
on HF.

### Qwen3-Embedding-0.6B — works

Loads cleanly, leaves ~3.9 GiB headroom. Embedding dim = 1024.

```
[loaded] in 11.6s, RSS=2.15 GB, avail=3.92 GB
[short]      dim=1024 latency mean=598ms   min=588ms
[medium]     dim=1024 latency mean=2352ms  min=2310ms
[long]       dim=1024 latency mean=13317ms min=13217ms
[batch8_med] latency=14011ms (1751ms/text)
```

Latencies (mean of 3, CPU-only):

| Input             | Latency |
|-------------------|---------|
| Short (~10 words) | 598 ms |
| Medium (~30 words)| 2.35 s |
| Long (~260 words) | 13.3 s |
| Batch of 8 medium | 14.0 s total (1.75 s/text) |

### jinaai/jina-embeddings-v3 — loads, but too slow

570M params. Loads cleanly in fp16, fits RAM with ~3.8 GiB to spare.
Inference latency on CPU is **20–60× slower than Qwen3-0.6B** because
Jina's custom `xlm-roberta-flash-implementation` modeling code is built
for GPU flash-attention and falls back to a slow path on CPU. Not viable
for our use case.

Needs `transformers<4.50` (newer transformers triggers
`AttributeError: 'XLMRobertaLoRA' object has no attribute 'all_tied_weights_keys'`).

```
[loaded] in 9.2s, RSS=2.47 GB, avail=3.77 GB
[short]   dim=1024 latency mean=19305ms min=18502ms
[medium]  dim=1024 latency mean=29022ms min=28023ms
[long]    dim=1024 latency mean=62644ms min=61680ms
```

| Input             | v3 fp16  | Qwen3-0.6B fp16 |
|-------------------|----------|-----------------|
| Short (~10 words) | 19.3 s   | 0.60 s |
| Medium (~30 words)| 29.0 s   | 2.35 s |
| Long (~260 words) | 62.6 s   | 13.3 s |

(batch8 was killed manually — extrapolating ≈ 4 min.)

### jinaai/jina-embeddings-v5-text-small-retrieval — works, but no clear win over Qwen3-0.6B

677M params, dim 1024, max seq 32k, Matryoshka dims 32–1024.
Newer transformers / torch (no pin on `<4.50` like v3). No `flash_attn`
config needed on CPU — the model loads with eager attention.

Use `prompt_name='query'` for queries and `prompt_name='document'` for
passages.

```
[pre]   RSS=0.83 GB, avail=5.50 GB
[loaded] in 20.4s, RSS=3.34 GB, avail=3.29 GB
[short]      dim=1024 latency mean=1086ms  min=534ms
[medium]     dim=1024 latency mean=2745ms  min=1753ms
[long]       dim=1024 latency mean=23333ms min=22219ms
[batch8_med] latency=25167ms (3146ms/text)
[final] RSS=3.27 GB, avail=2.84 GB
```

| Input             | v5-small-retrieval (min) | Qwen3-0.6B (min) |
|-------------------|--------------------------|------------------|
| Short (~10 words) | 534 ms                   | 588 ms |
| Medium (~30 words)| 1.75 s                   | 2.31 s |
| Long (~260 words) | 22.2 s                   | 13.2 s |
| Batch of 8 medium | 25.2 s (3.15 s/text)     | 14.0 s (1.75 s/text) |

Steady-state short/medium are roughly on par with Qwen3-0.6B, but long
inputs and batches are ~1.7–1.8× slower, and RSS is ~1.2 GB higher
(3.34 GB vs 2.15 GB). Not a clear upgrade on this box.

### jinaai/jina-embeddings-v5-text-nano-retrieval — fastest CPU embedder we've tested

239M params, dim 768. Loads cleanly, ~1.96 GB RSS. EuroBERT backbone
(eager attention, no GPU-specific code paths).

```
[pre]   RSS=0.83 GB, avail=5.46 GB
[loaded] in 10.4s, RSS=1.96 GB, avail=4.31 GB
[short]      dim=768 latency mean=164ms  min=115ms
[medium]     dim=768 latency mean=3436ms min=510ms
[long]       dim=768 latency mean=7583ms min=6056ms
[batch8_med] latency=6963ms (870ms/text)
[final] RSS=1.98 GB, avail=4.85 GB
```

| Input             | v5-nano-retrieval (min) | Qwen3-0.6B (min) |
|-------------------|-------------------------|------------------|
| Short (~10 words) | 115 ms                  | 588 ms |
| Medium (~30 words)| 510 ms                  | 2.31 s |
| Long (~260 words) | 6.06 s                  | 13.2 s |
| Batch of 8 medium | 7.0 s (870 ms/text)     | 14.0 s (1.75 s/text) |

Steady-state ~2–5× faster than Qwen3-0.6B across all sizes, ~0.2 GB
less RSS, and ~32k-token context. Trade-off: dim 768 instead of 1024
— retrieval quality should be checked against Qwen3-0.6B before
switching.

The `mean` latencies are heavily skewed by a slow first run (cold
torch graph / kernel autotune). `min` reflects steady-state.

### jinaai/jina-embeddings-v2-base-de — OOM at load

161M params (bilingual DE/EN). Custom `jina-bert-implementation` code
crashes the process with SIGKILL (`exit 137`) during
`SentenceTransformer(...)` construction, even with 5.7 GiB free. The
likely culprit is the eagerly-allocated ALiBi bias tensor for the
8192-token max sequence length. Not viable.

```
[pre]   RSS=0.68 GB, avail=5.67 GB
<silent SIGKILL>
EXIT=137
```

## Cross-encoder / reranker

### jinaai/jina-reranker-v2-base-multilingual — works well

278M params, multilingual. Loads in 6 s, RSS stays under 2 GB, latency
is usable for top-K reranking after a vector search.

Needs `sentence-transformers>=3,<4`, `transformers<4.50`, and the
constructor kwarg is `automodel_args` (not `model_kwargs`).

```
[loaded] in 6.2s, RSS=1.62 GB, avail=5.27 GB
[k=1]   latency mean=544ms   min=526ms   (544 ms/pair)
[k=8]   latency mean=3000ms  min=2990ms  (375 ms/pair)
[k=32]  latency mean=21018ms min=20219ms (657 ms/pair)
[final] RSS=1.72 GB, avail=5.13 GB
```

Sanity ranking against 8 mixed German passages for the query *"Wo finde
ich Informationen zur Stadtgeschichte Münsters?"* — Münster-related
passages cluster at the top, Berlin / Pizza / Python land at the bottom
with scores ~+0.02:

```
1. score=+0.612  Muenster hat eine lange Geschichte als Bischofsstadt ...
2. score=+0.583  Muenster ist eine Grossstadt in Nordrhein-Westfalen ...
3. score=+0.271  Der Aasee in Muenster ist ein beliebtes Naherholungsgebiet ...
4. score=+0.271  Die Westfaelische Wilhelms-Universitaet Muenster wurde 1780 ...
5. score=+0.239  Der Westfaelische Friede wurde 1648 in Muenster und Osnabrueck ...
6. score=+0.024  Berlin ist die Hauptstadt der Bundesrepublik Deutschland.
7. score=+0.023  Die Pizza Margherita stammt aus Neapel ...
8. score=+0.023  Python ist eine interpretierte Programmiersprache ...
```

## Takeaways

- **Hard RAM ceiling ~5.3 GiB** for any single model (no swap, ~2.5 GiB
  baseline).
- **No GPU**, so anything heavier than ~1B params in fp16 will either OOM
  or be too slow to be useful interactively.
- **Embedding:** **jinaai/jina-embeddings-v5-text-nano-retrieval** is
  the new front-runner — ~2–5× faster than Qwen3-Embedding-0.6B on CPU
  at steady state, ~0.2 GB less RSS, 32k context. Caveat: dim 768
  (vs 1024) and retrieval quality on our German Münster corpus is
  unverified — bench recall before switching. Qwen3-0.6B remains the
  safe fallback. v5-small-retrieval is not worth the extra RAM and
  longer-input latency on this box. v3 loads but is unusably slow on
  CPU (custom flash-attn path); v2-base-de OOM-kills at load.
- **Reranker:** **jinaai/jina-reranker-v2-base-multilingual** is the
  recommended cross-encoder. ~375 ms/pair when batching 8 candidates is
  fine for top-K reranking after vector search.
- **Pin transformers `<4.50`** for any Jina model with
  `trust_remote_code=True` — newer `transformers` broke the LoRA
  loader path (`all_tied_weights_keys`).
- Test scripts live in `~/embedding_test/` on the server:
  `test_jina_v3.py`, `test_jina_v5_small.py`, `test_jina_v5_nano.py`,
  `test_jina_reranker.py`, `test_jina_v2de.py` — each with a
  `# /// script` PEP 723 header so `uv run --no-project <file>.py` is
  enough.
- For generation models, anything we deploy will need **aggressive
  quantization** (e.g. llama.cpp Q4_K_M ≈ 2.4 GB) to fit alongside the
  embedding model and the FastAPI process.
