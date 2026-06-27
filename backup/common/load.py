"""Pure-Python load generator: a fallback for ``vllm bench serve``.

The hosted workshop image ships the ``vllm`` CLI, whose ``vllm bench serve``
drives concurrent load and reports throughput and latency percentiles. When
that CLI is not on PATH (for example, running these notebooks against a remote
endpoint from a laptop), this module reproduces the essential measurement with
the OpenAI client: fire ``num_prompts`` streaming requests at a fixed
concurrency, record TTFT and per-token latency for each, and sample the vLLM
KV-cache and queue gauges from /metrics while the load runs.

It returns the same fields the notebooks read from the CLI result, so the
sweep, the plots, and the analysis run unchanged in either environment.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor

from . import metrics
from .config import chat_extra

DEFAULT_PROMPT = "Write a detailed paragraph about GPU memory and how it is used."


def _percentile(values, q):
    """Nearest-rank percentile. Returns None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]


def _mean(values):
    return (sum(values) / len(values)) if values else None


def run_level(client, model, metrics_url, concurrency, num_prompts=None,
              output_len=128, prompt=DEFAULT_PROMPT, request_timeout=120):
    """Drive one concurrency level and return throughput + latency stats.

    Mirrors the fields ``vllm bench serve --save-result`` writes that the
    notebooks consume: output_throughput, request_throughput, mean/p95 TTFT,
    mean/p95 TPOT, plus peak KV usage, peak queue depth, and preemptions
    scraped from /metrics during the run.

    Each request carries a hard timeout and no client retries, so a stalled
    connection fails fast and is counted rather than hanging the whole sweep.
    """
    num_prompts = num_prompts or max(concurrency * 4, 16)
    # Per-request timeout, no retries: a load generator must never block forever.
    # with_options returns a fresh client COPY, which drops the build_client wrapper,
    # so we re-apply the per-model kwargs (Qwen3 thinking off) at the call site below.
    client = client.with_options(timeout=request_timeout, max_retries=0)
    extra = chat_extra(model)

    peak = {"kv": 0.0, "waiting": 0.0}

    def _preempt():
        try:
            return metrics.snapshot(metrics_url)["vllm:num_preemptions_total"]
        except Exception:
            return 0.0

    preempt_start = _preempt()
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                s = metrics.snapshot(metrics_url)
                peak["kv"] = max(peak["kv"], s["vllm:gpu_cache_usage_perc"])
                peak["waiting"] = max(peak["waiting"], s["vllm:num_requests_waiting"])
            except Exception:
                pass
            time.sleep(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    def fire_one(_):
        start = time.time()
        first = None
        toks = 0
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_len, temperature=0.0, stream=True,
                stream_options={"include_usage": True},
                **extra,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first is None:
                        first = time.time()
                    toks += 1
                if getattr(chunk, "usage", None):
                    toks = chunk.usage.completion_tokens or toks
        except Exception:
            return {"error": True}
        end = time.time()
        ttft = (first - start) if first else (end - start)
        gen = max(toks - 1, 1)
        tpot = ((end - first) / gen) if first else 0.0
        return {"ttft_ms": ttft * 1000, "tpot_ms": tpot * 1000, "tokens": toks}

    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        res = list(pool.map(fire_one, range(num_prompts)))
    wall = time.time() - wall_start

    stop.set()
    sampler.join(timeout=2)
    preempt_end = _preempt()

    ok = [r for r in res if not r.get("error")]
    total_tokens = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"]]

    return {
        "concurrency": concurrency,
        "output_throughput": (total_tokens / wall) if wall else 0.0,
        "request_throughput": (len(ok) / wall) if wall else 0.0,
        "mean_ttft_ms": _mean(ttfts),
        "p95_ttft_ms": _percentile(ttfts, 95),
        "mean_tpot_ms": _mean(tpots),
        "p95_tpot_ms": _percentile(tpots, 95),
        "peak_kv": peak["kv"],
        "peak_waiting": peak["waiting"],
        "preemptions": preempt_end - preempt_start,
        "failed": len(res) - len(ok),
    }
