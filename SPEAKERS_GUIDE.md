# Speaker's Guide

How to teach the foundation modules (1-4) and the optional capstone (9) of "Agents That
Own Their Inference." Du'An presents Modules 1-4 and the capstone; Omer presents the
optimization block, Modules 5-8. Each student runs on a dedicated GPU against a vLLM they
operate. Total foundation time is about 55 minutes; the capstone is optional and runs only
if time allows.

Read this alongside the notebooks. For each module: the frame, the points to land, the live
cells and what the room should see, the usual questions, rough timing, and what to do if the
GPU or model is slow.

---

## Module 1: The Inference Stack and Runtime Landscape

**Frame it.** Module 0 proved the endpoint is reachable. Module 1 makes the case for owning
that endpoint and gives the room the vocabulary for the rest of the day: prefill, decode,
time to first token, tokens per second, and the idea that the engine is a real layer.

**Points to land.**
- The notebook pod has no GPU. Every GPU operation happens in the vLLM pod, reached over HTTP.
  Show the architecture diagram and say it plainly.
- A request has two phases. Prefill is time to first token, decode is time per output token.
  Decode is the slow part and has a hard ceiling (the math is Module 2).
- The engine is a real layer. The same idea is fast or slow depending on the runtime and the
  card under it. Stress that the CPU-vs-vLLM cell is an illustration, not a benchmark.
- An agent is many requests. Its wall clock is the sum of its steps, so the slow step sets the
  pace. This is why the rest of the workshop is about inference.
- Owning has a cost crossover and three non-cost axes: data residency, rate limits, control.

**Live cells and what the room should see.**
- Section 2, one chat completion: a sentence from a server they control.
- Section 3, the streaming trace: TTFT in the low hundreds of milliseconds, decode at tens of
  tokens per second on the 4B.
- Section 4, the engine layer: the vLLM path finishing many times faster than the stated CPU
  baseline, even though the GPU model is larger. The CPU baseline is a representative figure the
  notebook prints; it loads no model, so it cannot crash the 1 GB pod. The point still lands.
- Section 6, the agent loop: four steps, each its own request, `prompt_tokens` climbing each
  step. Point at the climb and name it the context tax.
- Section 7, the cost cell: edit `requests_per_day` live to move the crossover.

**Usual questions.**
- "Why is the 4B faster than the 0.6B?" Because it runs on a GPU behind a serving engine and
  the small one runs on a CPU with a plain loop. Size did not make it fast.
- "Is this a real benchmark?" No. Say so before they ask. The fair comparison is Module 7.
- "Why does decode have a ceiling?" The GPU reads the weights once per token. Module 2 turns
  that into a number.

**Timing.** About 12 to 15 minutes. If you are short, run sections 2, 3, and 6 live, and talk
through 4, 7, and 8.

**If the GPU or model is slow.** The streaming trace still teaches the shape even with a high
TTFT. If the vLLM pod is not ready, run section 4's CPU path alone to make the engine-layer
point, and come back to the live cells once the pod is up.

---

## Module 2: Units and the Memory Budget

**Frame it.** Module 1 left them with a decode ceiling. Module 2 explains it. A model is files
you can size with arithmetic, and one card is a fixed budget split between weights and the KV
cache. This module is mostly arithmetic on real configs, so it runs even when the GPU is busy.

**Points to land.**
- Weight footprint is parameters times bytes per weight. The 4B is about 8 GB at BF16. That is
  fixed, paid before any request arrives.
- Precision is bytes per weight. FP8 drops the 4B to about 5 GB, roughly 35 percent smaller.
  Smaller weights decode faster and free memory for cache. Module 5 measures whether quality holds.
- The KV-cache memory formula: 2 x layers x kv_heads x head_dim x precision_bytes. For the 4B
  that is 147,456 bytes per token, about 144 KiB. Grouped-query attention makes it 4x smaller
  than full multi-head attention would.
- The KV pool is what is left after weights, scaled by gpu-memory-utilization. Pool tokens
  divided by sequence length is how many requests fit. That is the concurrency limit Omer
  pushes against in Module 7.

**Live cells and what the room should see.**
- Section 3, parameter counts: 0.6B near 1.5 GB, 4B near 8 GB at BF16, read from real metadata.
- Section 4, FP8: about 5.2 GB, 35 percent smaller.
- Section 5, decode ceiling: about 45 tok/s for the BF16 4B, about 69 for FP8.
- Section 6, the KV formula: 144 KiB per token and the 4x GQA factor, both from the real config.
- Section 7, the budget: a pool of tens of thousands of tokens and a concurrency table. If the
  server is reachable it reads the real `num_gpu_blocks`; if not, it computes the same budget.

**Usual questions.**
- "Why is the cache so small per token?" Grouped-query attention. Point at the 4x in section 6.
- "Why 0.7 utilization, not 0.9?" It leaves headroom and is the manifest default. Omer raises
  it in Module 8 and measures the effect.
- "Does FP8 hurt quality?" Maybe. That is the whole reason Module 5 measures before keeping it.

**Timing.** About 12 to 15 minutes. The arithmetic cells are fast. If short, run sections 3, 6,
and 7 and talk through the rest.

**If the GPU or model is slow.** This module barely needs the GPU. Config inspection uses the
  bundled `assets/` copies, so it needs no Hub access, and the budget arithmetic runs offline. Only
section 7's live pool read needs the server, and it falls back to the computed budget.

---

## Module 3: Prefill, Decode, and the KV Cache

**Frame it.** Module 2 sized the cache. Module 3 builds it and measures it. The numpy sections
earn the concept from first principles; the live sections show it on their own server. This is
the module that owns time to first token.

**Points to land.**
- Attention scores are a square matrix, one row and column per token, so they grow with the
  square of the sequence length. The keys and values you keep are linear. That split is the
  whole reason the KV cache exists.
- A request has two phases. Prefill is one parallel pass that fills the cache. Decode is
  sequential, one token at a time, each step reading all the weights and the growing cache.
- The KV cache trades memory for not recomputing the past. The numpy demo shows naive
  generation doing about N times more key/value work than cached, with the same answer.
- Resolve metric names at runtime. Show the resolved name on screen so the room sees the
  discipline.

**Live cells and what the room should see.**
- Section 4, numpy: naive 45,150 versus cached 300 key/value computes at N=300 (150x), and
  "same answer: True." This runs with no GPU, so it always works.
- Section 5, TTFT: client-side time to first token in the low hundreds of ms, decode at tens of
  tokens per second, then the server's own histogram and the resolved metric name.
- Section 6, the KV gauge: zero idle, a higher peak while a long request decodes, back toward
  zero after it finishes.
- Section 7, prefix cache: cold request with a low hit rate and higher TTFT, warm request with
  a high hit rate and a much lower TTFT. Run the two cells back to back.

**Usual questions.**
- "Why is decode slower than prefill?" Decode is sequential and reads all the weights per token.
  Prefill processes the prompt in parallel.
- "Does the warm speedup always happen?" Only with prefix caching on and the blocks not evicted.
  Point them at the precondition note in section 7.
- "Why resolve metric names?" vLLM renames them across versions. Hardcoding breaks silently.

**Timing.** About 15 to 18 minutes. The numpy sections are quick. If short, run sections 4, 5,
and 7 and describe section 6.

**If the GPU or model is slow.** Sections 2 to 4 are pure numpy and always run, so the core
concept lands regardless. For the live sections, a high TTFT still teaches the shape. If the
prefix-cache hit rate looks wrong, another tenant likely evicted the blocks; re-run the two
cells together.

---

## Module 4: Dense versus MoE on the GPU

**Frame it.** This is the last foundation module and the bridge to Omer. It turns the
memory-bound fact into a picture (the roofline) and uses it to explain MoE, the first
"read fewer bytes" technique. End by handing off to Omer's optimization block.

**Points to land.**
- The roofline has two limits: a flat compute line and a sloped memory line. They cross at the
  ridge, about 297 FLOP per byte for this card. Decode at batch 1 sits far down the memory side,
  reaching well under one percent of peak compute. Prefill sits near the compute line.
- Decode is memory-bound because one token reads every weight once. That is the whole reason the
  optimization block exists.
- A mixture-of-experts model stores every expert but reads only the active ones. Derived from the
  real Qwen3-30B-A3B config: 30.5B total, about 3.3B active per token, so it reads about nine
  times fewer bytes than a dense 30B and decodes about that much faster.
- Thinking models pay a reasoning tax. A long trace before the answer costs decode time on every
  agent step. Measurements run with thinking off; agents turn it on when they need to plan.

**Live cells and what the room should see.**
- Section 2, the roofline plot: ridge at 297, decode dot far down the memory side. Runs offline.
- Section 4, the MoE derivation: 30.5B total, 3.3B active, 9x fewer bytes per token, all derived
  from the config. Runs offline (Hub or bundled `assets/`).
- Section 5, decode rate: the served 4B near its memory-bound ceiling. Needs the server.
- Section 6, the reasoning tax: thinking off returns fast, thinking on emits a long trace and
  takes several times longer. Needs the server.

**Usual questions.**
- "Why store 30B to use 3B?" You pay VRAM footprint to save on the per-token read. The active
  read is what bandwidth caps, so MoE decodes fast for its quality.
- "Why not serve the MoE here?" Even at FP8 it overflows the 20 GB card. It stays a derivation.
- "Why 107 TFLOP/s, not 427?" The 427 is FP8 with 2:1 sparsity. Use the dense figure for the
  compute line.

**Timing.** About 12 to 15 minutes. The roofline and MoE derivation are the core and run offline.
If short, run sections 2 and 4 and talk through 5 and 6.

**If the GPU or model is slow.** The roofline and the MoE derivation need no GPU, so the
headline lands regardless. The two live measurements are illustrative; describe the expected
shape if the server is busy. This module ends the foundation, so leave time to hand off to Omer.

---

## Module 9: Deploy Your Agent on the Inference You Own (optional capstone)

**Frame it.** This is optional and runs only if there is time. It is the payoff: an agent on
the inference the room tuned all workshop. The headline is that an agent is just a client of a
model, and its speed is the inference layer's speed.

**Points to land.**
- An agent's wall-clock time is almost entirely inference. The tool and CPU work round to zero
  against the model calls. So making an agent faster means making inference faster.
- The agent is a plain Deployment in the student's namespace, reaching vLLM by the short Service
  name. No CRDs, no controller, no cluster admin.
- Fire several agents at once and they batch on the server: six finish in far less than six times
  one, and their traffic lands in the same metrics read since Module 3.
- Every answer comes from the student's own vLLM, on their GPU.

**Live cells and what the room should see.**
- Section 2, the wall-clock split: inference is essentially 100 percent of the agent's time.
- Section 4, deploy: the rollout reaching Ready and one `sa-agent` pod Running.
- Section 5, talk to it: a tactical in-scope answer and an honest out-of-scope redirect.
- Section 6, batching: six sessions finishing in far less than six times one, a peak of several
  concurrent requests running, and a bump in KV usage. The code prints the slowest session, so
  the room sees the real (sub-linear, not perfect) speedup.

**Usual questions.**
- "Is the agent doing real tool calls?" Section 2 simulates one tool call to time it; the
  deployed agent is chat-only. Adding a real tool is the stretch exercise.
- "Why is the slowest of six not as fast as one?" Decode is bandwidth-bound, so a larger batch
  shares each weight read but lowers per-sequence speed. Still far better than serial.
- "Does this need cluster admin?" No. A Deployment, Service, and ConfigMap in your namespace.

**Timing.** About 15 minutes if the cluster is responsive. It is the last thing on the agenda, so
treat it as a bonus. If time is short, run sections 2, 4, 5, and 6 and skip the stretch exercises.

**If the GPU or cluster is slow.** The wall-clock split in section 2 makes the core point even if
you skip the deployment. If the rollout is slow, talk through sections 4 to 6 using the
architecture diagram, then run the batching demo once the pod is Ready. If `kubectl` cannot reach
the cluster, this capstone becomes a read-through; the foundation modules stand on their own.
