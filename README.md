# Agents That Own Their Inference

**Building Production AI Agents on Dedicated GPUs**

Every production agent today rents its intelligence. You pay per token, you send customer data to someone else's servers, and you hope the provider does not rate-limit you during a launch. This workshop flips that. You get a dedicated GPU, operate the inference server yourself with vLLM, and learn to read and tune the layer your agent sits on top of.

The focus is not agent frameworks. It is the inference layer underneath them: prefill and decode, KV cache tradeoffs, continuous batching under real concurrency, vLLM metrics, quantization, speculative decoding, and the bottlenecks that only show up when you run the server yourself.

## Start Here

Two ways in. Pick yours:

- **At the Akamai workshop (Path A).** You have an access card. Open its URL, sign in with the password, and you land in JupyterLab with this repo cloned and a dedicated GPU and vLLM already running. Open `00_prerequisites/00_connect_and_verify.ipynb` and work the numbered folders in order. Nothing to install.
- **Running it yourself (Path B).** Bring any Kubernetes cluster with an NVIDIA GPU, or create one on Akamai Cloud with the terraform in [`infra/`](infra/). Deploy `manifests/vllm.yaml`, set the environment variables in the table below, then open Module 0. Full steps in [Path B: run it yourself](#path-b-run-it-yourself). The GPU node bills hourly.

The same notebooks run in both paths.

## Who This Is For

Engineers who already call an LLM API and want to understand what happens when they bring inference in-house. You are comfortable with Python and the basics of containers. No deep Kubernetes or vLLM background is assumed. The notebooks teach the concepts; the environment removes the setup.

## What You Will Be Able To Do

By the end you can:

1. Explain where a request spends its time inside an inference server: prefill vs decode, TTFT vs TPOT.
2. Size a model against a GPU by counting parameters, precision, VRAM, bandwidth, and KV cache.
3. Explain why decode is memory-bound and how dense, MoE, quantization, batching, and speculative decoding try to move the bottleneck.
4. Switch a vLLM deployment from BF16 to FP8 and measure the throughput difference.
5. Add speculative decoding, measure acceptance/throughput, and decide whether the draft path is worth keeping.
6. Drive load until batching, queueing, KV pressure, and saturation show up in the metrics.
7. Tune vLLM serving-policy knobs and keep or reject the change based on throughput and latency.
8. Deploy an agent on the inference endpoint you own and watch its traffic in the same metrics.

## Prerequisite Paths

The notebooks never provision infrastructure. They assume a vLLM endpoint already exists and read connection details from environment variables, so the same notebook runs in either path.

### Path A: Akamai-hosted workshop

You receive an access card. Open the URL, sign in with the password on the card, and you land in JupyterLab. Your namespace, kubeconfig, dedicated GPU, and deliberately under-tuned vLLM are already running. Module 0 verifies the connection and orients you.

### Path B: run it yourself

The one real requirement: a Kubernetes cluster with one NVIDIA GPU node, running the vLLM Deployment from `manifests/vllm.yaml`. Two ways to get there.

**Bring your own cluster.** Any Kubernetes with an NVIDIA GPU works. Two lines in `manifests/vllm.yaml` are Akamai-specific, and each carries a comment saying what to change: the storage class, and the `pool: gpu` node selector. Apply the manifest, then set the environment variables below.

**Start from zero with the terraform in [`infra/`](infra/).** You need an [Akamai Cloud account](http://login.linode.com/signup?promo=akm-dev-git-300-31126-M055) with a `LINODE_TOKEN` API token, plus terraform and kubectl installed. The GPU node is billed hourly at $0.52/hr, and the signup credit does not cover GPU plans, so tear the cluster down when you finish.

From the repo root:

```bash
cd infra
terraform init
terraform apply                    # cluster + GPU pool + NVIDIA gpu-operator, ~10 min

terraform output -raw kubeconfig | base64 -d > kubeconfig.yaml
export KUBECONFIG=$PWD/kubeconfig.yaml

kubectl apply -f ../manifests/vllm.yaml
kubectl rollout status deploy/vllm --timeout=20m   # first boot downloads the models into the PVC

kubectl port-forward svc/vllm 8000:8000            # leave this running
```

In a second terminal, from the repo root:

```bash
export KUBECONFIG=$PWD/infra/kubeconfig.yaml
export VLLM_HOST=http://localhost:8000/v1
export MODEL_NAME=Qwen/Qwen3-4B
pip install -r requirements.txt
jupyter lab
```

Open Module 0. The other variables in the table below keep their defaults.

**Tear it down when you finish.** Delete the workload first so the CSI driver removes the block-storage volume, then destroy the cluster:

```bash
kubectl delete -f manifests/vllm.yaml
cd infra && terraform destroy
```

**Modules 1 through 4 need only the endpoint.** They never touch the cluster. If you already run vLLM somewhere, set `VLLM_HOST` and `MODEL_NAME` and start the concepts half while your cluster provisions. Modules 0 and 5 through 9 need the cluster. The endpoint must be vLLM specifically: the measurement labs parse vLLM's own metric names, so Ollama or llama.cpp behind an OpenAI-compatible URL will not work.

What you need in either path: a running vLLM OpenAI-compatible endpoint, a small model on a single GPU, a kubeconfig with permission to edit the vLLM Deployment (the hosted platform hands you a namespace-scoped one; your own cluster's admin kubeconfig works as is), and Jupyter with this repo cloned.

## How To Start

1. Open `00_prerequisites/00_connect_and_verify.ipynb` and run it top to bottom. It confirms your endpoint, namespace, GPU-backed vLLM pod, and a working chat completion.
2. Work the numbered folders in order. Each module names the modules that come before it.
3. Modules 5 through 8 are Omer's performance block. Treat each technique as a hypothesis: make the change, run the same measurement, compare the result, and decide what configuration to keep.
4. Module 9 is Du'An's capstone: put an agent on top of the inference layer you measured.

If you are running Path B, install the Python dependencies first:

```bash
pip install -r requirements.txt
```

## Environment Variables

Every notebook reads these. In Path A they are already set. In Path B you set them yourself.

| Variable | What it is | Default |
|---|---|---|
| `VLLM_HOST` | Base URL of your vLLM OpenAI-compatible endpoint, including `/v1` | `http://vllm:8000/v1` |
| `MODEL_NAME` | The model id your vLLM server is serving | `Qwen/Qwen3-4B` |
| `NAMESPACE` | Your Kubernetes namespace | `default` |
| `KUBECONFIG` | Path to your namespace-scoped kubeconfig | unset, uses default context |
| `VLLM_API_KEY` | Only if your endpoint enforces a key | `not-needed` |

The vLLM Prometheus metrics URL is derived from `VLLM_HOST` automatically: the same host with `/metrics` instead of `/v1`.

## Agenda

| Module | Presenter | What you do | Time |
|---|---|---|---:|
| 0 Connect and verify | Du'An | Confirm endpoint, namespace, GPU-backed vLLM, and chat completion | 5 min |
| 1 Inference stack | Du'An | Compare renting vs owning and trace one request through vLLM | 12 min |
| 2 Units and memory budget | Du'An | Count parameters, precision, tokens, VRAM, bandwidth, and KV cache | 6 min |
| 3 Prefill, decode, KV cache | Du'An | Build attention/KV-cache intuition and watch live cache metrics | 20 min |
| 4 Dense vs MoE | Du'An | Use the roofline to explain dense vs MoE decode behavior | 15 min |
| 5 Quantization | Omer | Measure BF16, switch the shared vLLM manifest to FP8, compare throughput and quality risk | 25 min |
| 6 Speculative decoding | Omer | Add a 0.6B drafter, measure acceptance/throughput, and decide whether it applies here | 30 min |
| 7 Engine saturation | Omer | Drive concurrency until batching, queueing, and the saturation knee appear | 20 min |
| 8 Tune and evaluate | Omer | Try one vLLM knob, benchmark, and keep or reject the new operating point | 30 min |
| 9 Agent on Kubernetes | Du'An | Deploy an agent on the inference you own and watch its traffic in metrics | 12 min |

The full set runs longer than a single 120-minute session, so the live workshop runs the spine and treats deeper experiments as take-home work.

## Modules

- **`00_prerequisites/`** Get oriented and confirm everything is reachable.
- **`01_inference_stack/`** Compare inference runtimes, owning-vs-renting tradeoffs, and one request path.
- **`02_units_and_memory_budget/`** Count model memory, tokens, VRAM, bandwidth, and KV cache headroom.
- **`03_prefill_decode_kv_cache/`** Separate prefill from decode and derive the KV cache cost.
- **`04_dense_vs_moe/`** Explain dense vs MoE decode behavior with the roofline lens.
- **`05_quantization/`** Establish a BF16 baseline, switch the shared vLLM manifest to FP8, and compare throughput.
- **`06_speculative_decoding/`** Add draft-and-verify decoding with the 0.6B Qwen drafter and measure whether it helps or hurts.
- **`07_inference_engine_saturation/`** Connect engine internals, batching, queueing, and saturation metrics.
- **`08_tune_and_evaluate/`** Try one vLLM serving-policy change, benchmark the result, and choose a defensible operating point.
- **`09_optional_agents_on_k8s/`** Put an agent on the inference you measured, as a plain Deployment in your namespace.

## Repo Layout

```text
agents-that-own-their-inference/
├── README.md
├── COURSE_OUTLINE.md
├── requirements.txt
├── pyproject.toml
├── infra/                 Path B paved path: terraform for LKE + GPU pool + gpu-operator
├── common/
│   ├── config.py          reads VLLM_HOST, MODEL_NAME, NAMESPACE, KUBECONFIG
│   ├── metrics.py         scrapes and parses vLLM /metrics
│   ├── load.py            pure-Python load generator with metrics sampling
│   ├── loadtest.py        notebook-friendly concurrency sweeps
│   └── vllm_admin.py      scoped helpers for vLLM deployment edits
├── manifests/
│   └── vllm.yaml          shared vLLM deployment edited through Modules 5-8
└── NN_module_name/
    ├── NN_module_name.ipynb
    └── images/
```

## Open Source

Every tool in this workshop is open source: vLLM, the OpenAI-compatible Python client, Kubernetes, and the Qwen models. You can run all of it on hardware you control, with no vendor lock-in on the inference layer. That is the point: owning your inference means owning the stack underneath your agent.

## License

Apache-2.0.
