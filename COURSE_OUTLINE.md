# Agents That Own Their Inference — Course Outline

**Workshop:** Agents That Own Their Inference: Building Production AI Agents on Dedicated GPUs
**Venue:** AI Engineer World's Fair, 120-minute hands-on workshop
**Presenters:** Du'An Lightfoot (foundations + operational close), Khaja Omer (optimization)
**Format:** Jupyter notebooks, one dedicated GPU per attendee, run against a vLLM you operate
**Status:** Module 3 is built and verified end to end. It is the template every other module is held to.

---

## The bar

Every module follows the same arc, and nothing ships that does not clear it:

1. **Implement the mechanism** in numpy, small enough to read, real enough to run.
2. **Derive the numbers** from first principles: FLOPs, bytes, the ridge point, the acceptance rate.
3. **Measure it** on the GPU you own, confirming the theory and recovering hardware properties from observed behavior.

No "change a flag and watch a metric." A senior engineer should leave each module knowing something they could not get from a blog post.

## Structure

Three parts, two presenters. Du'An builds the machine and closes on the payoff; Khaja teaches the techniques that make it fast.

---

## Part 1 — The machine  ·  Du'An

Understand the inference stack from the transformer up.

### 1. The Inference Stack & Runtime Landscape
- **Frame:** the landscape of model runtimes (vLLM, SGLang, TensorRT-LLM, llama.cpp) and why this workshop runs vLLM; then vLLM's features and the value each brings (continuous batching, KV-cache management, kernel support).
- **Build:** trace one request end to end, client to server to engine to scheduler to KV cache to streamed tokens.
- **Derive:** the request lifecycle, where time goes (TTFT is prefill, TPOT is decode).
- **Measure:** connect to your vLLM, send a request, read the metrics that prove the lifecycle.
- *Source:* adapt `00_prerequisites` + `01_renting_vs_owning`. *Status:* adapt + add the landscape and the trace.
- *Sync:* Khaja proposed the runtime-landscape framing; Du'An agreed.

### 2. Transformer Deep Dive  —  CUT
Dropped per the sync. Du'An: "we dont need to cover this." This is an inference setup-and-use workshop, not model internals. The one piece that matters for inference (attention's role in prefill versus decode) is already in Module 3.

### 3. Prefill, Decode & the KV Cache  **[DONE, VERIFIED]**
- **Build:** attention by hand; naive generation at `O(n^2)`; the KV cache at `O(n)`, plotted side by side.
- **Derive:** `2 * layers * kv_heads * head_dim * bytes` per token; the GQA 4x; the budget-to-concurrency ceiling.
- **Measure:** the live KV gauge filling under load; recover your card's bandwidth from the decode rate (~320 GB/s); batching lifts decode 35 → ~1,800 tok/s.
- *Source:* rebuilds `02_inference_and_memory`. *Status:* `02_prefill_decode_kv_cache.ipynb`, verified against live `s01`. **This is the standard.**

### 4. Dense vs MoE on the GPU
- **Build:** run a dense model and a Mixture-of-Experts model (the catalog has `Qwen3-30B-A3B`) and compare their decode behavior on the same card.
- **Derive:** why MoE generates faster, it activates only a few experts per token, so it moves far fewer weight bytes than its total size (total versus active parameters); the roofline / arithmetic-intensity lens that explains the gap and why so many new models are MoE.
- **Measure:** decode tokens/s and the effective bytes moved per token, dense versus MoE, on your GPU.
- *Source:* roofline cell seeded in `04_saturate_your_gpu`. *Status:* reshape (MoE focus).
- *Sync:* Khaja proposed dense-vs-MoE over a GPU-architecture deep-dive (too dense for this audience); Du'An: MoE is important, it explains how models fit on GPUs and why they are faster. The roofline stays as the lens, not the headline.

> **Hand off to Khaja.**

---

## Part 2 — Make it fast  ·  Khaja

The optimization techniques that move the numbers.

### 5. Quantization
- **Build:** start on a non-quantized model, then switch to a quantized one and feel the difference; quantize a weight tensor by hand; then quantize the KV cache (for example fp8 / turbo KV) to serve more users per GPU.
- **Derive:** number formats (sign/exp/mantissa) and why FP8 keeps outlier range INT8 clips; bytes saved; the accuracy cost (Pi truncated to 3 is 4.5%, cubed 12.9%); freed memory to concurrent sequences.
- **Measure:** the non-quant vs quant performance gap; footprint drop; KV headroom and extra concurrent users from KV-cache quantization; accuracy on a small eval.
- *Source:* base in `06_quantization_with_llm_compressor`. *Status:* deepen.
- *Sync:* Khaja, start non-quant then quant to show the difference, add KV-cache quantization (turbo quant) for more users per GPU.

### 6. Attention Mechanisms (FlashAttention, and how to choose)
- **Frame:** a high-level overview of attention mechanisms (MHA, GQA, MQA, FlashAttention) and how to choose among them. The deep IO-complexity math is summarized, not derived; the audience does not need to implement a kernel.
- **Show:** why FlashAttention wins (it never materializes the `s x s` matrix, it is bound by HBM reads not FLOPs) and what the engine does for you.
- **Measure:** the attention-memory difference, naive versus tiled, as a demonstration.
- *Source:* none. *Status:* build fresh (overview altitude).
- *Sync:* Khaja, this is math-heavy; a high-level overview plus how-to-choose is the right altitude, not a kernel derivation.

### 7. Speculative Decoding
- **Build:** draft-and-verify with a small draft model or MTP; the acceptance check, so you see which proposed tokens are kept.
- **Derive:** acceptance rate, expected accepted tokens per verify step, the speedup formula, when it pays.
- **Measure:** acceptance rate and end-to-end speedup on real prompts; the latency-throughput tradeoff.
- *Reference:* Khaja's example `akamai-developers/speculative-decoding-example-vllm-blackwell`. Note: it targets a Blackwell GPU; the workshop card is an RTX 4000 Ada on vLLM v0.20.2, so confirm it runs on the workshop hardware.
- *Source:* none. *Status:* build fresh. **Needs platform work (see Platform support).**

### 8. Continuous Batching & Saturation
- **Build:** drive rising concurrency and watch continuous batching pack requests into one forward pass.
- **Derive:** Little's Law and the saturation knee, tied to the roofline from Module 4.
- **Measure:** the batch forming, then the queue, KV pressure, and preemption past the knee.
- *Source:* base in `03_serving_with_vllm` + `04_saturate_your_gpu`. *Status:* reuse.

> **Hand back to Du'An.**

---

## Part 3 — Operate it  ·  Du'An

Run it under real load on the GPU you own.

### 9. Tune the Engine
- **Build:** edit the engine flags in the vLLM manifest and redeploy through Kubernetes.
- **Derive:** the `gpu-memory-utilization` byte budget and the batch-cap arithmetic, predicting the gain before measuring.
- **Measure:** the before-and-after sweep, throughput up while latency holds, proven against the metrics.
- *Source:* base in `05_optimize_the_server` + `08_benchmark_and_evaluate`. *Status:* reuse + deepen.

### 10. The Agent on Inference You Own
- **Build:** deploy a small agent into your namespace, pointed at the vLLM you tuned.
- **Derive:** an agent as a plain client of a private endpoint; where scale-to-zero fits.
- **Measure:** the agent's traffic landing in the same metrics you tuned, closing the loop.
- *Source:* base in `09_optional_agents_on_k8s`. *Status:* reuse. The capstone.

---

## Mapping from the current repo

| Existing folder | Becomes | What happens |
|---|---|---|
| `00_prerequisites` | Module 1 | adapt (on-ramp + request trace) |
| `01_renting_vs_owning` | cold open / framing | becomes the intro, not a numbered module |
| `02_inference_and_memory` | Module 3 | rebuilt deep (done) |
| `03_serving_with_vllm` | Module 8 | reuse |
| `04_saturate_your_gpu` | Module 8 (+ seeds 4) | reuse; roofline seed to Module 4 |
| `05_optimize_the_server` | Module 9 | reuse |
| `06_quantization_…` | Module 5 | deepen |
| `07_two_models_one_gpu` | — | dropped (time-slicing killed) |
| `08_benchmark_and_evaluate` | Module 9 | folds in |
| `09_optional_agents_on_k8s` | Module 10 | reuse |

Of 10 folders: 8 map forward, `01` becomes framing, `07` is dropped. Of the new modules: 1 done (M3), 1 cut (M2, Transformer), 5 reuse a base (M1, M5, M8, M9, M10), and 3 fresh builds (M4 reshaped to MoE, M6 overview, M7).

## Platform support (verified)

Du'An's modules (1, 3, 4, 9, 10) run on the platform as configured, on the current `Qwen/Qwen3-4B`. No platform changes. Module 4's MoE comparison wants an MoE model served (the catalog has `Qwen3-30B-A3B`), so confirm that plan fits one student GPU.

Khaja's half has the only real platform work:
- **Speculative Decoding (the gap).** `Qwen3.5-4B` is not in the `sizing.py` catalog, and there is no spec-decode wiring anywhere (no `--speculative-config`, MTP, draft-model, or EAGLE flags). vLLM is pinned at `v0.20.2`. Path: add `Qwen3.5-4B` to `sizing.py` with the MTP spec-decode args in its `vllm_args` (the dedicated vLLM template already ranges over `vllm_extra_args`), then confirm v0.20.2 runs Qwen3.5 MTP.
- **Quantization:** serving quantized models is supported; the quantize-by-hand step needs a GPU the notebook pod lacks (same as today's `06`).
- **FlashAttention / Continuous Batching:** no platform work; supported by vLLM today.

## Open decisions for the 4pm sync

1. **The live 120-minute cut.** Even split, it is more than 120 minutes hands-on. Suggested live path: Du'An opens with Module 1 (brief) → 3 → 4; Khaja teaches a subset of 5–8; Du'An closes with Module 10. The rest is take-home depth.
2. **Confirm Modules 9 and 10 are Du'An's** (Khaja's set is 5–8).
3. **numpy vs a GPU framework** for the mechanism modules, and the final teaching order.
