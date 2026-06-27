# common/ helpers

Shared code the notebooks import. All of it reads connection details from the
environment (`VLLM_HOST`, `MODEL_NAME`, `NAMESPACE`, `KUBECONFIG`), so the same cell runs
in the hosted workshop and in a bring-your-own setup.

| File | What it gives the student |
|---|---|
| `config.py` | `get_settings()`, `build_client()`: resolve the endpoint and an OpenAI client |
| `metrics.py` | low-level `/metrics` parse used inside the notebooks (`snapshot`, `parse_histogram_avg`) |
| `ttft_metrics.py` | `vllm_stats()` and `watch()`: read TTFT, TPOT, throughput, KV, queue inline, no Grafana |
| `loadtest.py` | `run()`, `sweep()`, `background()`: stress concurrency and token volume |
| `vllm_admin.py` | `switch_model()`, `current_model()`: swap the served model via kubectl |
| `load.py` | the older per-level load runner the saturate notebook uses; `loadtest.py` is the student-facing one |

## The uniform stress-and-watch cell

This is the one pattern every load lab uses. Drive load in the background, watch the
metrics move, then let the block end and the load stops.

```python
from common import loadtest, ttft_metrics

with loadtest.background(concurrency=32, input_tokens=512, output_tokens=128):
    ttft_metrics.watch(30)        # prints and plots TTFT, KV %, waiting, throughput
```

Raise `concurrency` to fill the batch and the queue. Raise `input_tokens` to fill the KV
cache. The triage from the metrics catalog reads straight off the plot: if `waiting`
climbs while `KV %` sits near 100, you are out of KV cache, so lower `--max-model-len` or
scale out.

## The saturation sweep

One summary row per concurrency level, no background thread.

```python
from common import loadtest
loadtest.sweep([1, 8, 32, 64, 128])   # throughput climbs, then TTFT p95 turns up at the knee
```

## Switch the model

vLLM serves one model per process. Switching restarts the pod and loads from the PVC
cache in seconds.

```python
from common import vllm_admin
vllm_admin.AVAILABLE_MODELS          # served target models, not draft models
vllm_admin.switch_model("RedHatAI/Qwen3-4B-FP8-dynamic")
```

## Where each piece is used

- Module 1, 3: light reads. One streamed request for TTFT and TPOT, the KV gauge under a
  few requests.
- Module 4: one live decode-rate read plus a bandwidth estimate; the 0.6B model stays
  reserved for Module 6 speculative decoding.
- Module 7 (Omer): the full `sweep()` plus `watch()` to drive saturation and read the knee.
- Module 8 (Omer): `watch()` before and after a `--gpu-memory-utilization` or
  `--max-num-seqs` change, to prove the tune.

## Metrics the watcher surfaces

From the workshop catalog, mapped to what `vllm_stats()` returns:

- Latency histograms: `time_to_first_token_seconds` (TTFT), `inter_token_latency_seconds`
  or `time_per_output_token_seconds` (TPOT), `e2e_request_latency_seconds`,
  `request_queue_time_seconds`.
- Throughput counters: `generation_tokens_total`, `prompt_tokens_total` as a per-second rate.
- Scheduler gauges: `num_requests_running`, `num_requests_waiting`, `num_requests_swapped`,
  `num_preemptions_total`.
- Cache gauges: `kv_cache_usage_perc` (falls back to `gpu_cache_usage_perc` on older vLLM).
