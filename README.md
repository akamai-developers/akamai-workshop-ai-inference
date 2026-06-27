# Agents That Own Their Inference

**Building Production AI Agents on Dedicated GPUs**

Every production agent today rents its intelligence. You pay per token, you send customer data to someone else's servers, and you hope the provider does not rate-limit you during a launch. This workshop flips that. You get a dedicated GPU, you operate the inference server yourself with vLLM, and you learn to read and tune the layer your agent sits on top of.

The focus is not agent frameworks. It is the inference layer underneath them: continuous batching under real concurrency, KV cache tradeoffs, vLLM metrics, quantization, and the bottlenecks that only show up when you run the server yourself.

## Start here

Two ways in. Pick yours:

- **At the Akamai workshop (Path A).** You have an access card. Open its URL, sign in with the password, and you land in JupyterLab with this repo cloned and a dedicated GPU and vLLM already running. Open `00_prerequisites/00_connect_and_verify.ipynb` and work the numbered folders in order. Nothing to install.
- **Running it yourself (Path B).** Stand up the infrastructure with the separate [`akamai-workshop-platform`](https://github.com/akamai-developers/akamai-workshop-platform) repo (the own-inference workshop), install `requirements.txt`, set the environment variables in the table below, then open Module 0.

The same notebooks run in both. Details for each path are in [The two prerequisite paths](#the-two-prerequisite-paths).

## Who this is for

Engineers who already call an LLM API and want to understand what happens when they bring inference in-house. You are comfortable with Python and the basics of containers. No deep Kubernetes or vLLM background is assumed. The notebooks teach the concepts; the environment removes the setup.

## What you will be able to do

By the end you can:

1. Explain where a request spends its time inside an inference server (prefill vs decode, TTFT vs TPOT) and point to the vLLM metric that proves it.
2. Serve a model with vLLM and watch it handle many concurrent requests through continuous batching and PagedAttention.
3. Drive concurrent load at a deployment you control until KV cache pressure, queueing, and preemption show up in the metrics.
4. Tune the server through Kubernetes to lift throughput while holding per-request latency.
5. Switch from BF16 to FP8 and measure the performance gain, while knowing where workload evals fit in production.
6. Explain when speculative decoding helps, how to measure it, and where it can backfire.
7. Drive load until batching, queueing, and saturation show up in the metrics.
8. Tune vLLM knobs and weigh speed, latency, cost, and quality against each other.

## The two prerequisite paths

The notebooks never provision infrastructure. They assume a vLLM endpoint already exists and read connection details from environment variables, so the same notebook runs in either path.

### Path A: at the Akamai-hosted workshop (default)

You receive an access card. Open the URL, sign in with the password on the card, and you land in JupyterLab. Your namespace, your kubeconfig, your dedicated GPU, and a deliberately under-tuned vLLM are already running. Module 0 verifies the connection and orients you. Nothing to install, no account to create.

### Path B: bring your own environment (self-paced)

You run the same modules against your own Akamai Cloud account or any Kubernetes cluster with a GPU. You stand up the infrastructure first using the separate `akamai-workshop-platform` repo, then point the notebooks at your endpoint by setting the environment variables below.

What you need in either path: a running vLLM OpenAI-compatible endpoint, a small model on a single GPU, a kubeconfig scoped to your namespace with permission to edit your vLLM deployment, and JupyterLab with this repo cloned.

## How to start

1. Open `00_prerequisites/00_connect_and_verify.ipynb` and run it top to bottom. It confirms your endpoint, your namespace, and a working chat completion.
2. Work the numbered folders in order. Each module names the modules that come before it.
3. Modules 5 through 8 are the performance block. The live session covers the core path and leaves deeper experiments as take-home work.

If you are running Path B, install the Python dependencies first:

```bash
pip install -r requirements.txt
```

## Environment variables

Every notebook reads these. In Path A they are already set. In Path B you set them yourself.

| Variable | What it is | Default |
|---|---|---|
| `VLLM_HOST` | Base URL of your vLLM OpenAI-compatible endpoint, including `/v1` | `http://vllm:8000/v1` |
| `MODEL_NAME` | The model id your vLLM server is serving | `Qwen/Qwen3-4B` |
| `NAMESPACE` | Your Kubernetes namespace | `default` |
| `KUBECONFIG` | Path to your namespace-scoped kubeconfig | unset (uses default context) |
| `VLLM_API_KEY` | Only if your endpoint enforces a key | `not-needed` |

The vLLM Prometheus metrics URL is derived from `VLLM_HOST` automatically (the same host, `/metrics` instead of `/v1`).

## Agenda

| Module | What you do | Time |
|---|---|---|
| 0 Prerequisites | Connect and verify your environment | 5 min |
| 1 Renting vs owning | Compare a hosted API call against your self-hosted vLLM | 7 min |
| 2 Inference and memory | Measure TTFT and inter-token latency; read the KV cache gauge | 15 min |
| 3 Serving with vLLM | Watch continuous batching form under light concurrency | 15 min |
| 4 Saturate your GPU | Drive load until queueing, KV pressure, and preemption appear | 15 min |
| 5 Quantization | Measure BF16, edit the shared vLLM manifest to FP8, and compare throughput | 25 min |
| 6 Speculative decoding | Add a 0.6B draft model with `--speculative-config` and compare throughput | 30 min |
| 7 Engine mechanics and saturation | Connect attention kernels, batching, and GPU saturation metrics | 20 min |
| 8 Tune and evaluate | Change vLLM knobs, benchmark the result, and pick an operating point | 30 min |
| 9 Agents on Kubernetes | Deploy an agent on the inference you tuned and talk to it | 12 min |

The full set runs longer than a single 120-minute session, so the live workshop runs the spine and uses Modules 5 through 8 as the performance block. The deeper experiments and optional agent capstone are take-home material when time is tight.

## Modules

- **`00_prerequisites/`** Get oriented and confirm everything is reachable.
- **`01_renting_vs_owning/`** The case for operating your own inference server.
- **`02_inference_and_memory/`** Where a request spends its time and memory.
- **`03_serving_with_vllm/`** Continuous batching under light concurrency.
- **`04_saturate_your_gpu/`** Push to the bottleneck and name it.
- **`05_quantization/`** Establish a BF16 performance baseline, switch the shared vLLM manifest to FP8, and compare throughput.
- **`06_speculative_decoding/`** Add draft-and-verify decoding with the 0.6B Qwen drafter and measure whether it helps.
- **`07_inference_engine_saturation/`** Connect engine internals, batching, and saturation metrics.
- **`08_tune_and_evaluate/`** Tune vLLM, benchmark the result, and choose a defensible operating point.
- **`09_optional_agents_on_k8s/`** Put an agent on the inference you tuned, as a plain Deployment in your namespace. The payoff.

## Repo layout

```
agents-that-own-their-inference/
├── README.md
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── common/
│   ├── config.py     reads VLLM_HOST, MODEL_NAME, NAMESPACE, KUBECONFIG
│   ├── metrics.py    scrape and parse vLLM /metrics, plot a time series
│   └── load.py       pure-Python load generator, a fallback for the vllm CLI
└── NN_module_name/
    ├── NN_module_name.ipynb
    └── images/       the branded architecture diagram for the module
└── manifests/
    └── vllm.yaml     the shared vLLM deployment edited through Modules 5-8
```

## Open source

Every tool in this workshop is open source: vLLM, GuideLLM, and the Qwen models. You can run all of it on hardware you control, with no vendor lock-in on the inference layer. That is the point. Owning your inference means owning the stack underneath your agent, and these open source projects are what make that practical today.

## License

Apache-2.0.
