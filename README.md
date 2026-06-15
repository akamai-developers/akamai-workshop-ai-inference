# Agents That Own Their Inference

**Building Production AI Agents on Dedicated GPUs**

Every production agent today rents its intelligence. You pay per token, you send customer data to someone else's servers, and you hope the provider does not rate-limit you during a launch. This workshop flips that. You get a dedicated GPU, you operate the inference server yourself with vLLM, and you learn to read and tune the layer your agent sits on top of.

The focus is not agent frameworks. It is the inference layer underneath them: continuous batching under real concurrency, KV cache tradeoffs, vLLM metrics, quantization, and the bottlenecks that only show up when you run the server yourself.

## Who this is for

Engineers who already call an LLM API and want to understand what happens when they bring inference in-house. You are comfortable with Python and the basics of containers. No deep Kubernetes or vLLM background is assumed. The notebooks teach the concepts; the environment removes the setup.

## What you will be able to do

By the end you can:

1. Explain where a request spends its time inside an inference server (prefill vs decode, TTFT vs TPOT) and point to the vLLM metric that proves it.
2. Serve a model with vLLM and watch it handle many concurrent requests through continuous batching and PagedAttention.
3. Drive concurrent load at a deployment you control until KV cache pressure, queueing, and preemption show up in the metrics.
4. Tune the server through Kubernetes to lift throughput while holding per-request latency.
5. Quantize a model with LLM Compressor to shrink its memory footprint, then measure the accuracy tradeoff.
6. Benchmark a deployment and weigh speed, cost, and accuracy against each other with real numbers.
7. Put an agent on top of the inference you tuned, running on a GPU you own.

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
3. Modules 6 and 8 are the depth you take home. The live session samples them and points here for the full treatment.

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
| 5 Optimize the server | Tune the engine flags through Kubernetes and prove the gain | 30 min |
| 6 Quantization | Quantize with LLM Compressor and measure the accuracy tradeoff | 20 min |
| 7 Two models, one GPU | Run two models on one card and route between them | 12 min |
| 8 Benchmark and evaluate | Sweep speed, cost, and accuracy; pick an operating point | 20 min |
| 9 Agents on Kubernetes | Deploy an agent with kagent on the inference you tuned | 12 min |

The full set runs longer than a single 120-minute session, so the live workshop runs the spine (Modules 0 through 5, plus 7 or 9) and leaves Modules 6 and 8 as the take-home depth.

## Modules

- **`00_prerequisites/`** Get oriented and confirm everything is reachable.
- **`01_renting_vs_owning/`** The case for operating your own inference server.
- **`02_inference_and_memory/`** Where a request spends its time and memory.
- **`03_serving_with_vllm/`** Continuous batching under light concurrency.
- **`04_saturate_your_gpu/`** Push to the bottleneck and name it.
- **`05_optimize_the_server/`** Edit the manifest, redeploy, prove the gain. The heart of the workshop.
- **`06_quantization_with_llm_compressor/`** Shrink the footprint, measure the cost in quality.
- **`07_two_models_one_gpu/`** Two models sharing a card, with routing.
- **`08_benchmark_and_evaluate/`** Speed, cost, and accuracy together, with a recommended operating point.
- **`09_optional_agents_on_k8s/`** Put an agent on the inference you tuned, with kagent. The payoff.

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
    ├── images/       the branded architecture diagram for the module
    └── manifests/    where a module edits Kubernetes manifests
```

## Open source

Every tool in this workshop is open source: vLLM, LLM Compressor, GuideLLM, kagent, and the Qwen models. You can run all of it on hardware you control, with no vendor lock-in on the inference layer. That is the point. Owning your inference means owning the stack underneath your agent, and these open source projects are what make that practical today.

## License

Apache-2.0.
