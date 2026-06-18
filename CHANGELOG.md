# Changelog

All notable changes to this workshop are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Rewrote Module 9 to drop kagent and the Discord bridge. The agent is now a plain
  Kubernetes Deployment in your own namespace: a small HTTP service (`agent/agent.py`,
  the openai client as its only dependency, mounted from a ConfigMap) plus a Service.
  You apply the manifest, port-forward, and talk to it in scope and out of scope to see
  the system-prompt guardrails hold, then watch vLLM `/metrics` move as it answers. No
  CRDs, no controller, no helm, nothing cluster-scoped, so a scoped kubeconfig can run
  it. Closes on where scale-to-zero fits (KServe/Knative for replicas, the autoscaler
  for nodes).
- `common/config.py` reads the platform's native `VLLM_HOST`, `MODEL_NAME`, and
  `VLLM_API_KEY` directly, so the workshop drops into the deployed environment with no
  variable translation. Added namespace resolution (env, then the active kube-context,
  then `default`).
- Trimmed `requirements.txt` to the five packages the notebooks import. Dropped
  `llmcompressor` (it pulls torch and OOMs the 1 Gi notebook pod; Module 6 installs it
  on demand behind a guard) and the unused `prometheus-client`. Replaced the leftover
  `.env.example` with this workshop's variables.

### Fixed
- Set the workshop model to the hybrid `Qwen/Qwen3-4B` and wired thinking end to end.
  It reasons (emits `<think>`) when asked, which Module 9's agent needs to reason before
  it picks a tool. `vllm-baseline.yaml` now serves it with `--reasoning-parser deepseek_r1`
  alongside `--tool-call-parser hermes`, so the thinking is split out of the answer and
  the tool-call JSON. NOTE: the platform's own-inference preset must serve this same model
  id; its catalog default points at the non-thinking `Qwen/Qwen3-4B-Instruct-2507`, and a
  mismatch makes Module 5's apply and the Module 9 agent 404. Notebooks read
  `settings.model_name`, so they follow whatever the platform injects.
- Thinking is on where it helps and off where it does not. The measurement labs send
  `enable_thinking=false` (via `build_client` / `chat_extra`) so benchmarks count answer
  tokens, not a `<think>` block; `agent/agent.py` sends `enable_thinking=true` so the
  agent reasons. Both still drop an empty `tools=[]`, which vLLM 0.20+ rejects.
- `common/load.py` re-applies the per-model kwargs at the call site. `run_level` copies
  the client with `with_options(...)` for a per-request timeout, which drops the
  `build_client` wrapper, so the pure-Python load path would otherwise lose the Qwen3
  handling. Correct for any thinking Qwen3 and honest about what the wrapper guarantees.
- `09_optional_agents_on_k8s`: corrected a note that claimed the platform NetworkPolicy
  already lets `app: agent` reach vLLM. It does not; the agent is a new client the
  workspace-to-vLLM rule does not cover, so the note now states the rule the cluster
  needs for the capstone to connect.
- Baseline `--gpu-memory-utilization` raised from `0.40` to `0.50`. On a 20 GB card a
  4B model's weights leave `0.40` with no room for KV cache blocks, so vLLM fails to
  start; `0.50` is the working under-tuned floor the module raises from. Updated the
  manifest, its README, and the Module 5 notebook.
- Added `HF_HUB_DISABLE_XET=1` to the vLLM pods. The newer xet downloader has stalled
  at 0 B/s on some clusters, which looks like a pod stuck pulling weights with no error.
  Harmless if xet is not in play.
- `00_prerequisites`: replaced `kubectl get nodes` (Forbidden under the scoped
  per-student RBAC) with a namespace pod listing that shows the node and GPU.
- `06_quantization_with_llm_compressor`: a `RUN_QUANTIZE` guard so the module reads and
  runs cleanly when the notebook pod has no GPU, with the quantize cells opt-in.
- `07_two_models_one_gpu`: the notebook now deploys `two-models.yaml` (scaling the
  baseline `vllm` to zero first) and cleans up after, instead of assuming both servers
  are already running.

## [1.1.0] - 2026-06-15

### Changed
- Rewrote all ten notebooks in the hand-written `Module N` workshop voice: an intro that recalls the previous module's gap, learning objectives, a references line, a branded architecture diagram, numbered explainer-then-code sections, a fail-first beat shown on screen, a `Things to know` section, a `Try it yourself` section, a summary, and a next-module pointer. No em-dashes; every notebook ships clean with no outputs.
- Reframed Module 9 from an optional victory lap to the capstone: the agent runs on the inference you tuned. Updated `README.md` and `OUTLINE.md` terminology from "lab"/"section" to "module" to match.

### Added
- A branded architecture diagram per module under `NN_*/images/`.
- `common/load.py`: a pure-Python concurrent load generator that mirrors the fields of `vllm bench serve`, used automatically by Modules 4, 5, and 8 when the `vllm` CLI is not on PATH.

### Fixed
- `common/metrics.py`: `snapshot()` now returns the KV cache gauge under both `vllm:gpu_cache_usage_perc` and `vllm:kv_cache_usage_perc`. vLLM's V1 engine renamed the metric, so without this the KV readings silently showed zero on newer servers.
- `05_optimize_the_server/manifests/vllm-baseline.yaml`: `strategy: Recreate` (a single-GPU pod cannot roll-update without deadlock), `enableServiceLinks: false` (a Service named `vllm` injects `VLLM_PORT`, which vLLM misreads as its own port), and tool-calling flags so the same server can back the Module 9 agent. The notebook's `set_flag()` now preserves the YAML list marker when it patches a value.
- `07_two_models_one_gpu/manifests/two-models.yaml`: `enableServiceLinks: false` and `strategy: Recreate` on both deployments.
- `09`: the `ModelConfig` baseUrl now uses the cross-namespace FQDN, a `NetworkPolicy` allows kagent to reach vLLM through the default-deny boundary, the system-prompt YAML block scalar is correctly indented, and the invoke step uses the current kagent CLI (`--agent`/`--task` with a controller port-forward and an agent-ready wait).

### Removed
- `akamai-cloud-mcp-PROMPT.md`: a stray multi-session build prompt for an unrelated project that did not belong in the workshop repo.

## [1.0.0] - 2026-06-13

First complete release. All ten sections are built, validated, and pass the
prose and infrastructure checks. The repo is ready to publish.

### Added
- Phase 10, section 09 `09_optional_agents_on_k8s.ipynb`: the optional victory lap. Installs kagent with helm (CRDs first, then the controller), creates a `ModelConfig` pointing at the in-cluster vLLM Service and an `Agent` themed as an Akamai Cloud solutions architect with scope guardrails, tests it with `kagent invoke`, and documents the Discord bridge as a take-home with the conference caveat. Targets `kagent.dev/v1alpha2`.
- Phase 10 final QA pass: every notebook validates against nbformat and converts to a script, no em-dashes in generated content, no hardcoded infrastructure, the next-lab cross-links form a complete chain, and the README agenda matches the folders on disk.
- Phase 9, section 08 `08_benchmark_and_evaluate.ipynb`: a full `vllm bench serve` sweep, a throughput-vs-latency plot with an SLO line, a cost-per-million-tokens table from the instance hourly price, a small accuracy eval set, and a printed recommended operating point combining speed, cost, and quality. Notes GuideLLM for richer constant-rate sweeps.
- Phase 8, section 07 `07_two_models_one_gpu.ipynb` plus `manifests/`: a `two-models.yaml` running `vllm-fast` and `vllm-smart` on one GPU (each capped at 0.45 memory) with a manifests README on GPU time-slicing. The notebook builds a client per backend, runs a content-based Python router, drives both models concurrently, and reads each backend's KV cache gauge to show contention. agentgateway noted as the production routing layer.
- Phase 7, section 06 `06_quantization_with_llm_compressor.ipynb`: quantizes the model to FP8 with LLM Compressor `oneshot()` + `QuantizationModifier` (FP8_DYNAMIC), saves compressed-tensors, reports the footprint drop, shows how to serve the quantized model with vLLM, runs a small fixed-prompt quality check, reasons about the KV cache headroom gained, and notes the W4A16 path.
- Phase 6, section 05 `05_optimize_the_server.ipynb` plus `manifests/`: a deliberately under-tuned baseline vLLM Deployment and Service (`vllm-baseline.yaml`) and a manifests README explaining the four flags. The notebook runs the measure, read, change, redeploy, re-measure loop: baseline sweep, raise `--gpu-memory-utilization`, `kubectl apply` + `rollout restart` + `rollout status`, re-sweep, then raise the batch caps and sweep again, with a before/after throughput plot and guidance on reading metrics to pick the next change.
- Phase 5, section 04 `04_saturate_your_gpu.ipynb`: sweeps `vllm bench serve` across rising concurrency (1, 4, 16, 64, 128), samples KV cache and preemptions from `/metrics` during each level, plots the throughput and TTFT knee, names the bottleneck, and includes a pure-Python load fallback for environments without the vllm CLI.
- Phase 4, section 03 `03_serving_with_vllm.ipynb`: runs the same workload sequentially then concurrently, samples `/metrics` in a background thread to watch `vllm:num_requests_running` form the batch, plots it, and reports the throughput speedup from continuous batching.
- Phase 3, section 02 `02_inference_and_memory.ipynb`: measures TTFT and TPOT from a non-streamed and a streamed request, reads server-side TTFT/TPOT histograms and the KV cache gauges from `/metrics`, and explains prefill vs decode and PagedAttention.
- Phase 2, section 00 `00_connect_and_verify.ipynb`: prints resolved settings, lists namespace pods with `kubectl`, checks for a GPU on a node, and sends a first chat completion through the self-hosted endpoint. Documents Path A and Path B inline.
- Phase 2, section 01 `01_renting_vs_owning.ipynb`: same prompt against the self-hosted vLLM and an optional hosted API, a monthly hosted-cost estimate, and the data residency, rate limit, and control story.
- Phase 1 foundation: repository tree with one numbered folder per section, each with an `images/` placeholder; `manifests/` folders for sections 05 and 07.
- `common/config.py`: reads `VLLM_HOST`, `MODEL_NAME`, `NAMESPACE`, `KUBECONFIG`, and `VLLM_API_KEY` from the environment, builds an OpenAI client, and prints resolved settings without leaking secrets.
- `common/metrics.py`: fetch and parse the vLLM `/metrics` endpoint, read the gauges and counters the labs use, compute histogram averages for TTFT and TPOT, and a `SeriesRecorder` to sample and plot a metric over time.
- `requirements.txt` and `pyproject.toml` with the notebook dependencies. `vllm` and `guidellm` are noted as environment-provided.
- `README.md`: premise, audience, the two prerequisite paths, the agenda, environment variables, the section list, and an open source note.
