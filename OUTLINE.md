# Agents That Own Their Inference - Content Outline

**Workshop:** Agents That Own Their Inference: Building Production AI Agents on Dedicated GPUs
**Event:** AI Engineer World's Fair, San Francisco
**Format:** 120-minute hands-on workshop, delivered as a Jupyter notebook repo
**Maintainer:** Du'An Lightfoot, Senior Developer Advocate, Akamai
**Status:** content outline (drives the autonomous build in `PROMPT.md`)

This document is the source of truth for what the repo teaches and how it is organized. The infrastructure that hosts the modules lives in a separate repo (`akamai-workshop-platform`) and is out of scope here.

---

## Premise

Every production agent today rents its intelligence. You pay per token, you send customer data to someone else's servers, and you hope the provider does not rate-limit you during a launch. This workshop flips that: you get a dedicated GPU, you operate the inference server yourself, and you learn to read and tune the layer the agent sits on top of. The focus is not agent frameworks. It is the inference layer underneath them: continuous batching under real concurrency, KV cache tradeoffs, vLLM metrics, quantization, and the bottlenecks that only show up when you run the server yourself.

## Who this is for

Engineers who already call an LLM API and want to understand what happens when they bring inference in-house. Comfortable with Python and the basics of containers. No deep Kubernetes or vLLM background assumed. The notebooks teach the concepts; the provided environment removes the setup.

## Learning objectives

By the end, a participant can:

1. Explain where a request spends its time inside an inference server (prefill vs decode, TTFT vs TPOT) and point to the vLLM metric that proves it.
2. Serve a model with vLLM and watch it handle many concurrent requests through continuous batching and PagedAttention.
3. Drive concurrent load at a deployment they control until KV cache pressure, queueing, and preemption appear in the metrics.
4. Tune the server through Kubernetes (the engine flags) to lift throughput while holding per-request latency.
5. Apply quantization with LLM Compressor to shrink a model's memory footprint, then measure the accuracy tradeoff.
6. Benchmark a deployment and weigh speed, cost, and accuracy against each other with real numbers.

---

## Prerequisites and the two paths

The notebooks assume the inference environment already exists. They never provision infrastructure. Each module reads its connection details (the vLLM endpoint, the namespace, the kubeconfig) from environment variables, so the same notebook runs in either path below.

**Path A - At the Akamai-hosted workshop (default).**
You receive an access card. Open the URL, sign in with the password on the card, and you land in JupyterLab. Your namespace, your kubeconfig, your dedicated GPU, and a deliberately under-tuned vLLM are already running. Module 0 only verifies the connection and orients you. Nothing to install, no account to create.

**Path B - Bring your own environment (self-paced after the event).**
You run the same modules against your own Akamai Cloud account or any Kubernetes cluster with a GPU. You stand up the infrastructure first using the separate `akamai-workshop-platform` repo, then point the notebooks at your endpoint. The setup steps for Path B live in that repo and in `00_prerequisites`, not in the module content.

What you need in either path: a running vLLM OpenAI-compatible endpoint, a small model on a single GPU, a kubeconfig scoped to your namespace with permission to edit your vLLM deployment, and JupyterLab with the repo cloned.

---

## How the source course material maps in

The 9-lesson "efficient LLM deployment" course outline is a strong backbone. It supplies the fundamentals, the quantization module, and the benchmarking discipline. We keep that spine and add the parts that make this workshop different: the agent angle, the dedicated GPU, and the Kubernetes-native deploy-and-tune loop where students change the server themselves.

| Source lesson | Where it lands here |
|---|---|
| Introduction | `README.md` + `00_prerequisites` |
| Why Efficient LLM Deployment Matters | `01_renting_vs_owning` |
| Inference & Memory Fundamentals | `02_inference_and_memory` |
| LLM Optimization Fundamentals | introduced in `03_serving_with_vllm`, applied in `05_optimize_the_server` |
| Optimizing a Model with LLM Compressor | `06_quantization_with_llm_compressor` |
| Serving LLMs Efficiently with vLLM, Part I | `03_serving_with_vllm` |
| Serving LLMs Efficiently with vLLM, Part II | `04_saturate_your_gpu` + `05_optimize_the_server` |
| Measuring What Matters: Benchmarking and Evaluation | `08_benchmark_and_evaluate` |
| (new) Operate and tune through Kubernetes | `05_optimize_the_server` |
| (new) Two models on one GPU + routing | `07_two_models_one_gpu` |
| (new) Agents on Kubernetes | `09_optional_agents_on_k8s` |

The three stated learning bullets all fit: quantization plus accuracy tradeoff is Module 6, concurrent serving with continuous batching and PagedAttention is Modules 2 through 4, and benchmarking plus quality measurement is Module 8.

---

## Repository structure

Each module is its own folder with one notebook and the assets that module needs, following the Amazon Bedrock workshop pattern. Notebooks are environment-agnostic and read connection details from env vars.

```
agents-that-own-their-inference/
├── README.md                       # the why, the agenda, the two prereq paths, how to start
├── requirements.txt                # python deps for the notebooks (openai, requests, plotting, llmcompressor, etc.)
├── pyproject.toml                  # optional uv-managed dependency file
├── common/
│   ├── config.py                   # reads VLLM_HOST, MODEL_NAME, NAMESPACE, KUBECONFIG from env with sane defaults
│   └── metrics.py                  # helper to scrape and parse vLLM /metrics, plot live
├── 00_prerequisites/
│   ├── 00_connect_and_verify.ipynb
│   └── images/
├── 01_renting_vs_owning/
│   ├── 01_renting_vs_owning.ipynb
│   └── images/
├── 02_inference_and_memory/
│   ├── 02_inference_and_memory.ipynb
│   └── images/
├── 03_serving_with_vllm/
│   ├── 03_serving_with_vllm.ipynb
│   └── images/
├── 04_saturate_your_gpu/
│   ├── 04_saturate_your_gpu.ipynb
│   └── images/
├── 05_optimize_the_server/
│   ├── 05_optimize_the_server.ipynb
│   ├── manifests/                  # the under-tuned baseline vllm manifest students edit
│   └── images/
├── 06_quantization_with_llm_compressor/
│   ├── 06_quantization_with_llm_compressor.ipynb
│   └── images/
├── 07_two_models_one_gpu/
│   ├── 07_two_models_one_gpu.ipynb
│   ├── manifests/
│   └── images/
├── 08_benchmark_and_evaluate/
│   ├── 08_benchmark_and_evaluate.ipynb
│   └── images/
└── 09_optional_agents_on_k8s/
    ├── 09_optional_agents_on_k8s.ipynb
    └── images/
```

---

## Module-by-module breakdown

Each entry lists the objective, what the participant does, the key concepts, an estimated self-paced time, and whether it is on the live 120-minute path.

### 00 - Prerequisites: connect and verify (5 min, live)
- **Objective:** get oriented in the provided environment and confirm everything is reachable.
- **Do:** run a config cell that prints the vLLM endpoint, model name, and namespace; `kubectl get pods` in your namespace; a one-line chat completion; a GPU check.
- **Concepts:** the two paths, where your endpoint comes from, what a kubeconfig scopes you to.
- **Live path:** yes, as the on-ramp.

### 01 - Renting vs owning your inference (7 min, live)
- **Objective:** make the case for operating your own inference server.
- **Do:** call a hosted API and your self-hosted vLLM with the same prompt; compare a request that leaves your network against one that stays inside it; read the cost and rate-limit story.
- **Concepts:** per-token economics, data residency, rate limits, control.
- **Live path:** yes, the cold open.

### 02 - Inference and memory fundamentals (15 min, live)
- **Objective:** understand where a request spends its time and memory.
- **Do:** send one request, then a streamed one; measure time to first token and inter-token latency; scrape `/metrics` and read the KV cache gauges.
- **Concepts:** prefill is compute-bound and sets TTFT; decode is bandwidth-bound and sets TPOT; PagedAttention stores the KV cache in fixed 16-token blocks.
- **Live path:** yes.

### 03 - Serving LLMs efficiently with vLLM (15 min, live)
- **Objective:** see continuous batching form under light concurrency.
- **Do:** fire a handful of requests at once; watch `num_requests_running` and the batch grow; observe throughput climb as the batch fills.
- **Concepts:** continuous (in-flight) batching, the scheduler's decode-then-prefill step, why batching raises throughput.
- **Live path:** yes.

### 04 - Saturate your GPU (15 min, live)
- **Objective:** drive load until the server hits its limit and you can name the bottleneck.
- **Do:** run `vllm bench serve` at rising concurrency (1, 4, 16, 64, 128) against your endpoint; plot throughput and TTFT; push until `num_requests_waiting` grows, `kv_cache_usage_perc` approaches 1.0, and `num_preemptions_total` starts climbing.
- **Concepts:** queueing, KV cache pressure, recompute-based preemption in vLLM V1, the throughput-vs-latency knee.
- **Live path:** yes, the first climax.

### 05 - Optimize the server through Kubernetes (30 min, live)
- **Objective:** lift throughput while holding latency by tuning the engine, not the model.
- **Do:** open the baseline vLLM manifest (deliberately under-tuned); change `--gpu-memory-utilization`, `--max-num-seqs`, `--max-num-batched-tokens`, and `--max-model-len`; `kubectl apply` and `rollout restart`; re-run the load test; compare before and after across two or three iterations.
- **Concepts:** each flag's tradeoff, why most engine args require a restart, reading the metrics to decide the next change.
- **Live path:** yes, the heart of the session. Protect this block.

### 06 - Quantization with LLM Compressor (20 min, self-paced; sampled live)
- **Objective:** shrink the model's memory footprint and measure what it costs in quality.
- **Do:** quantize a small model with LLM Compressor `oneshot()` to FP8 or W4A16; save in compressed-tensors format; serve it with vLLM; compare KV cache headroom and throughput against the full-precision model; run a short quality check on a fixed prompt set to see the accuracy tradeoff.
- **Concepts:** weight vs activation quantization, FP8 and INT4 schemes, the memory-for-accuracy trade, why a smaller footprint means more KV cache and higher concurrency.
- **Live path:** sampled. The live session uses FP8 as one optimize lever in Module 5 and points to this module for the full treatment.

### 07 - Two models on one GPU and model routing (12 min, optional live)
- **Objective:** run two small models on a single card and route between them.
- **Do:** start a second vLLM process on the same GPU with a split memory budget, or place agentgateway in front of two model backends; send `model: "fast"` and `model: "smart"`; watch the two models contend for the same KV cache under load.
- **Concepts:** one model per vLLM process, memory splitting, content-based routing, contention as a first-class signal.
- **Live path:** optional, demo-or-do.

### 08 - Benchmark and evaluate (20 min, self-paced; sampled live)
- **Objective:** measure speed, cost, and accuracy together and make a defensible tradeoff.
- **Do:** run a structured sweep with `vllm bench serve` or GuideLLM; build throughput and latency curves; estimate cost per million tokens at a target latency; run a small evaluation set to score quality; write down the recommended operating point.
- **Concepts:** the difference between a benchmark number and a production SLO, choosing an operating point, cost per token at a latency target.
- **Live path:** sampled in Module 4; full version is self-paced.

### 09 - Deploy your agent on the inference you own (12 min, the capstone)
- **Objective:** put an agent on top of the inference you just tuned, deploy it to your namespace with kagent, and talk to it from your phone.
- **Do:**
  1. Install kagent with helm, CRDs first, then the controller, following the reference lab.
  2. Create a `ModelConfig` that points at YOUR in-cluster vLLM Service, then an `Agent` CRD that uses it. Run `kagent invoke` to test.
  3. Connect a Discord bridge so you can chat with the agent from your phone, reusing the `nba-discord-agent` pattern.
- **Agent theme (recommended): an Akamai Cloud helper / solutions-architect agent** that answers questions about Akamai Cloud and your cluster. It is on brand, it reinforces the thesis (an agent running on inference you own), and it is more reusable than a trivia bot. The `nba-discord-agent` is the ready-made alternative if you want a lower-build-risk demo. The agent's brain is the kagent CRD pointed at your vLLM; the Discord bridge is reused glue.
- **Concepts:** kagent `Agent` and `ModelConfig` CRDs, pointing an agent at a private vLLM endpoint, optional tools via `RemoteMCPServer`, the Discord bridge.
- **Conference caveat:** a Discord bot per student needs a bot token and a Discord app, which is setup tax in a live room. Default to `kagent invoke` or a simple phone-reachable chat at your `sNN` URL during the session, and document Discord as the take-home extension. If you want Discord live, pre-create bot tokens and hand them out on the access card.
- **Live path:** the payoff. Run it if time allows; it is where the whole workshop lands, with the inference layer still the lesson underneath.

---

## Live 120-minute path

The repo is a self-paced superset. The live session walks a curated path and points to the rest for later.

| Live segment | Module |
|---|---|
| 0:00 cold open | 01 |
| 0:07 connect | 00 |
| 0:18 anatomy | 02 |
| 0:33 batching to saturation | 03 + 04 |
| 0:48 optimize | 05 (FP8 lever borrows from 06) |
| 1:18 two models | 07 |
| 1:30 wrap and checklist | summary cell |
| 1:40 stretch | 09 |

Modules 6 and 8 are the depth the attendees take home.

---

## Tech stack and core dependencies

- Python with the `openai` client pointed at the vLLM OpenAI-compatible endpoint.
- `requests` and a Prometheus text parser for scraping vLLM `/metrics`.
- `matplotlib` or `plotly` for live throughput and latency plots.
- `vllm` CLI for `vllm bench serve` (load generation already whitelisted by the platform NetworkPolicy).
- `llmcompressor` for the quantization module.
- `kubectl` available in the JupyterLab terminal, with a namespace-scoped kubeconfig.
- A small model that fits one 20 GB card, for example Qwen3-4B or a 1.5B to 3B model when splitting the GPU.

---

## Resources

**vLLM**
- Production metrics reference: https://docs.vllm.ai/en/latest/design/metrics/
- Optimization and tuning guide: https://docs.vllm.ai/en/stable/configuration/optimization/
- Anatomy of vLLM (batching, PagedAttention, preemption): https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- Benchmarking CLI (`vllm bench serve`): https://docs.vllm.ai/en/latest/benchmarking/cli/
- vLLM docs home: https://docs.vllm.ai/

**Quantization**
- LLM Compressor on GitHub: https://github.com/vllm-project/llm-compressor
- LLM Compressor docs: https://docs.vllm.ai/projects/llm-compressor
- compressed-tensors format: https://github.com/neuralmagic/compressed-tensors

**Load and evaluation**
- GuideLLM benchmarker: https://github.com/vllm-project/guidellm
- GuideLLM walkthrough: https://developers.redhat.com/articles/2025/06/20/guidellm-evaluate-llm-deployments-real-world-inference
- Reference single-GPU vLLM benchmarks: https://www.databasemart.com/blog/vllm-gpu-benchmark-a6000

**Agents and serving (Module 9)**
- kagent: https://kagent.dev/ and https://github.com/kagent-dev/kagent
- Reference kagent install lab (CRDs first, then controller; ModelConfig points at in-cluster vLLM): https://labeveryday.github.io/learn-k8s/07-kagent/lab-01-install-kagent/
- Discord bridge pattern (instructor's working example): https://github.com/labeveryday/nba-discord-agent
- kserve: https://kserve.github.io/website/
- agentgateway: https://agentgateway.dev/

**Instructor-provided references (to fold in)**
- vLLM docs home (primary technical source): https://docs.vllm.ai/en/latest/
- Reference book on efficient LLM deployment (instructor-provided via Google Drive). Not yet accessible to the build. Add the title and a public or quotable source, or export the relevant chapters, so specific sections can be cited in Modules 2, 4, 6, and 8.

**Akamai Cloud (Path B and the hosted environment)**
- GPU compute instances: https://techdocs.akamai.com/cloud-computing/docs/gpu-compute-instances
- LKE getting started: https://techdocs.akamai.com/cloud-computing/docs/getting-started-with-lke-linode-kubernetes-engine
- The infrastructure repo (separate): `akamai-workshop-platform`

**Models**
- Qwen3 on Hugging Face: https://huggingface.co/Qwen
- Llama 3.1 8B: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

## Voice and style

Write in Du'An's Akamai Developer Advocate voice: developer to developer, tactical, concrete, and honest about tradeoffs. Lead with what the reader will do and why it matters, then show it. Open source is treated as a first-class path, not an afterthought.

Hard style rules, enforced with the writestat MCP on every markdown cell:
- No em-dashes. Use commas, parentheses, colons, or a rewrite.
- No AI tells or filler ("delve", "in today's fast-paced world", "it's worth noting", "as we can see").
- Short, direct sentences. Active voice. Second person for instructions.
- Every code cell has a one-line purpose above it and a "what you should see" note below it.
- Use the `tech-docs` skill conventions for structure and headings.
