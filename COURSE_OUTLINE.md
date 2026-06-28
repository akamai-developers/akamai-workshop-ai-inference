# Agents That Own Their Inference — Course Outline

**Workshop:** Agents That Own Their Inference: Building Production AI Agents on Dedicated GPUs
**Venue:** AI Engineer World's Fair, 120-minute hands-on workshop
**Presenters:** Du'An Lightfoot (Modules 0-4 and 9), Khaja Omer (Modules 5-8)
**Format:** Jupyter notebooks, one dedicated GPU per attendee, run against a vLLM you operate
**Status:** Modules 0-9 are implemented. The rehearsal cluster has run every notebook end to end.

---

## The Story

The workshop starts with a working inference endpoint, builds the mental model for why serving is expensive, then walks through the performance levers and closes by deploying an agent on the inference layer the attendee operated.

The teaching arc is:

1. **Own the endpoint.** Verify the vLLM server, Kubernetes namespace, and GPU-backed model.
2. **Understand the machine.** Count memory, decode cost, KV cache, dense/MoE behavior, and the roofline.
3. **Test serving-stack techniques.** Quantize, test speculative decoding, find saturation, and choose which changes deserve to stay.
4. **Put an agent on top.** Deploy a namespaced agent and watch its traffic land in the same metrics.

The important performance lesson is not "every technique makes this workload faster." The important lesson is controlled measurement. FP8 quantization showed a clear win in rehearsal. Draft-model speculative decoding did not win in this one-GPU setup, so Module 6 teaches how to recognize and explain a negative result, while linking to successful implementations where the drafter or method is a better fit. Module 8 closes the block by turning those measurements into a keep-or-reject operating decision.

## Presenter Flow

- **Du'An, Modules 0-4:** foundations, operating context, memory/KV/cache/roofline intuition, and handoff to optimization.
- **Omer, Modules 5-8:** performance block: candidate optimization techniques, speculative decoding tradeoffs, saturation, and measured operating-point selection.
- **Du'An, Module 9:** capstone: deploy an agent on the inference layer Omer just measured and tuned.

---

## Module Outline

### Module 0 — Connect and Verify Your Environment

- **Build:** resolve environment settings, list namespace pods, inspect the vLLM pod, and send a first OpenAI-compatible request.
- **Measure:** confirm `/v1/models` and a real chat completion from the vLLM endpoint.
- **Outcome:** attendees know their endpoint, namespace, kubeconfig, and GPU-backed vLLM pod are working.
- **Folder:** `00_prerequisites/`

### Module 1 — The Inference Stack and Runtime Landscape

- **Build:** point a small agent-style wrapper at the vLLM endpoint and trace one request end to end.
- **Derive:** where time goes: prefill, decode, TTFT, TPOT, and the runtime/scheduler/KV-cache layers.
- **Measure:** compare owning against renting with token counts, cost, data path, rate limits, and control.
- **Outcome:** attendees understand why owning the inference server matters and why this workshop uses vLLM.
- **Folder:** `01_inference_stack/`

### Module 2 — Units and the Memory Budget

- **Build:** compute model weight size from parameters and precision.
- **Derive:** bytes per number, tokens as the serving unit, VRAM/bandwidth, and the weights/KV/overhead memory split.
- **Measure:** compare FP16/BF16, FP8, and INT4 budgets to see how smaller weights leave more KV cache.
- **Outcome:** attendees can size a model against a GPU card before touching Kubernetes.
- **Folder:** `02_units_and_memory_budget/`

### Module 3 — Prefill, Decode, and the KV Cache

- **Build:** attention by hand, naive generation, and cached generation.
- **Derive:** why naive decode grows quadratically and KV-cache decode is linear; compute per-token KV cache cost.
- **Measure:** watch the live vLLM cache gauge move and connect prefill/decode to arithmetic intensity.
- **Outcome:** attendees understand the trade: spend GPU memory to avoid recomputing prior context.
- **Folder:** `03_prefill_decode_kv_cache/`

### Module 4 — Dense vs MoE on the GPU

- **Build:** compare dense and Mixture-of-Experts behavior through the roofline lens.
- **Derive:** why decode is memory-bound and why MoE can generate faster than its total parameter count suggests.
- **Measure:** connect active parameters, bytes moved per token, and decode throughput.
- **Outcome:** attendees see the first "read fewer bytes" performance technique and hand off to Omer for the measurement-driven optimization block.
- **Folder:** `04_dense_vs_moe/`

> **Hand off to Omer.**

### Module 5 — Quantization

- **Build:** measure the BF16 baseline, edit `manifests/vllm.yaml`, deploy `RedHatAI/Qwen3-4B-FP8-dynamic`, and confirm `/v1/models`.
- **Derive:** number formats, bytes moved per decode token, and why FP8 can preserve useful range while shrinking weight reads.
- **Measure:** compare BF16 versus FP8 throughput and TTFT with the same workload shape.
- **Outcome:** attendees decide, from measurement, that FP8 is the right baseline for the rest of this workshop's performance block.
- **Folder:** `05_quantization/`

### Module 6 — Speculative Decoding

- **Build:** add `--speculative-config` with `RedHatAI/Qwen3-0.6B-FP8-dynamic` as the draft model.
- **Derive:** accepted tokens per target step, speculation depth, acceptance rate, and draft overhead.
- **Measure:** compare the FP8 baseline against speculative decoding with the same workload shape and validate that the live deployment contains the speculative config.
- **Outcome:** attendees learn that speculative decoding can help substantially when the drafter/method fits, but can hurt when acceptance is low, depth is too high, scheduler tradeoffs dominate, or the drafter contends with the target on one GPU.
- **Folder:** `06_speculative_decoding/`

### Module 7 — Engine Mechanics and Saturation

- **Build:** drive rising concurrency against the measured deployment.
- **Derive:** continuous batching, PagedAttention, queueing, KV pressure, and the saturation knee.
- **Measure:** find the first point where throughput flattens while TTFT, waiting, or cache pressure rises.
- **Outcome:** attendees know the capacity boundary before they try to tune anything.
- **Folder:** `07_inference_engine_saturation/`

### Module 8 — Tune and Evaluate

- **Build:** try one vLLM serving-policy flag in the shared manifest and redeploy.
- **Derive:** how `gpu-memory-utilization`, `max-model-len`, `max-num-seqs`, and `max-num-batched-tokens` move cache headroom and scheduler behavior.
- **Measure:** rerun the same sweep and keep or reject the change based on throughput and latency.
- **Outcome:** attendees learn the operating discipline: Module 8 may keep the baseline, because tuning is a measured keep-or-reject loop, not a guaranteed bigger number.
- **Folder:** `08_tune_and_evaluate/`

> **Hand back to Du'An.**

### Module 9 — Deploy Your Agent on the Inference You Own

- **Build:** deploy a small Akamai Cloud solutions architect agent as a namespaced Kubernetes Deployment and Service.
- **Derive:** an agent is a client of a model endpoint; the important question is where that endpoint lives and who operates it.
- **Measure:** send concurrent agent questions and watch the traffic show up in vLLM metrics.
- **Outcome:** attendees close the loop: the agent runs on inference they own, measured, and operated.
- **Folder:** `09_optional_agents_on_k8s/`

---

## Live 120-Minute Cut

The full set is longer than 120 minutes if every cell and optional experiment is run live. Suggested live path:

1. Du'An: Modules 0-1 quickly, then Modules 2-4 selectively for the mental model.
2. Omer: Modules 5-8 as the performance decision block, emphasizing that FP8 wins here, speculative decoding may not, and every serving change needs evidence before it stays.
3. Du'An: Module 9 as the capstone if time allows, or as take-home material.

## Platform Assumptions

The rehearsal deployment used:

- one workspace/student
- dedicated per-student vLLM
- scoped kubeconfig
- one CPU node and one single-GPU node
- `Qwen/Qwen3-4B` as the initial model
- `RedHatAI/Qwen3-4B-FP8-dynamic` as the FP8 target
- `RedHatAI/Qwen3-0.6B-FP8-dynamic` as the speculative drafter
- shared `manifests/vllm.yaml` edited by Modules 5-8

Important platform/content notes:

- The shared manifest must start from `Qwen/Qwen3-4B` with no speculative config so manual walkthroughs begin cleanly.
- Module 5 switches the served model to FP8.
- Module 6 adds the drafter only through `--speculative-config`; the 0.6B model is never a served target model.
- Module 9 creates `sa-agent` with its own selector so it does not collide with any platform-created `agent` deployment.
