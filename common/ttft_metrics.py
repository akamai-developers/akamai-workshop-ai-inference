"""
ttft_metrics.py — read vLLM metrics in the notebook, no Grafana needed.

It scrapes the student's OWN vLLM Prometheus endpoint (http://vllm:8000/metrics). The
workshop NetworkPolicy already allows workspace to vllm:8000, so this works with zero
extra infra: no Prometheus, no Grafana, no public endpoint, no auth. It scales to 200
students because each one reads only their own pod.

vLLM reports latency as histograms (..._sum / ..._count). The averages below are the
DELTA of sum and count between two scrapes, which is the mean over the window you
measured. That is what you want when comparing models or tuning flags. Counters use a
per-second rate over the window. Gauges are read at the end of the window.

Metric names follow the workshop catalog. Where vLLM has renamed a metric across
versions, both names are tried and the first present wins.
"""
import os
import re
import time
import urllib.request

# Workspace gets VLLM_HOST=http://vllm:8000/v1 ; metrics live at /metrics (strip /v1).
_BASE = os.environ.get("VLLM_HOST", "http://vllm:8000/v1").split("/v1")[0]
METRICS_URL = f"{_BASE}/metrics"

# Histograms reported as a window average (sum/count delta). First base with samples wins.
_HIST = {
    "ttft_s":  ["vllm:time_to_first_token_seconds"],
    "tpot_s":  ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"],
    "e2e_s":   ["vllm:e2e_request_latency_seconds"],
    "queue_s": ["vllm:request_queue_time_seconds"],
}
# Counters reported as a per-second rate over the window.
_COUNTER = {
    "gen_tokens":    ["vllm:generation_tokens_total"],
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "preemptions":   ["vllm:num_preemptions_total"],
}
# Gauges read at the end of the window.
_GAUGE = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "swapped": ["vllm:num_requests_swapped"],
    "kv_pct":  ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}

_LINE = re.compile(r"(\S+?)(?:\{.*\})?\s+([0-9eE.+-]+)$")


def _parse(text: str) -> dict:
    """Sum each metric family across its label series into a flat dict. Testable."""
    acc: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            acc[m.group(1)] = acc.get(m.group(1), 0.0) + float(m.group(2))
        except ValueError:
            pass
    return acc


def _scrape() -> dict:
    """Fetch and parse the student's own /metrics endpoint."""
    text = urllib.request.urlopen(METRICS_URL, timeout=5).read().decode()
    return _parse(text)


def _first(d: dict, bases, suffix=""):
    """Return the value of the first base name present (with optional suffix), or None."""
    for base in bases:
        key = base + suffix
        if key in d:
            return d[key]
    return None


def vllm_stats(window_s: float = 5.0) -> dict:
    """Measure recent averages and rates over `window_s` seconds.

    Run inference (or drive load) during the window, then read the result. Pair it with
    loadtest.background() to watch the numbers move under concurrency.
    """
    a = _scrape()
    time.sleep(window_s)
    b = _scrape()

    out = {}
    for label, bases in _HIST.items():
        val = None
        for base in bases:
            dc = (b.get(f"{base}_count", 0) or 0) - (a.get(f"{base}_count", 0) or 0)
            ds = (b.get(f"{base}_sum", 0) or 0) - (a.get(f"{base}_sum", 0) or 0)
            if dc > 0:
                val = round(ds / dc, 4)
                break
        out[label] = val
    for label, bases in _COUNTER.items():
        rate = ((_first(b, bases) or 0) - (_first(a, bases) or 0)) / window_s
        out[label + "_per_s"] = round(rate, 1)
    for label, bases in _GAUGE.items():
        out[label] = _first(b, bases) or 0.0
    return out


def watch(duration_s: int = 60, step_s: float = 5.0, plot: bool = True):
    """Sample the key signals over time, print a line per step, and plot.

    Returns (timestamps, series) where series is a dict of lists. Drive load in another
    thread (loadtest.background) so the queue and the cache actually move.
    """
    ts = []
    series = {"ttft_ms": [], "tpot_ms": [], "throughput": [], "kv_pct": [], "waiting": []}
    elapsed = 0.0
    while elapsed < duration_s:
        s = vllm_stats(window_s=step_s)
        ttft_ms = None if s["ttft_s"] is None else round(s["ttft_s"] * 1000, 1)
        tpot_ms = None if s["tpot_s"] is None else round(s["tpot_s"] * 1000, 1)
        ts.append(round(elapsed, 1))
        series["ttft_ms"].append(ttft_ms)
        series["tpot_ms"].append(tpot_ms)
        series["throughput"].append(s["gen_tokens_per_s"])
        series["kv_pct"].append(round(s["kv_pct"] * 100, 1))
        series["waiting"].append(s["waiting"])
        print(f"t={elapsed:6.1f}s  TTFT={ttft_ms}ms  TPOT={tpot_ms}ms  "
              f"gen={s['gen_tokens_per_s']} tok/s  run={s['running']:.0f} "
              f"wait={s['waiting']:.0f}  KV={series['kv_pct'][-1]}%  preempt/s={s['preemptions_per_s']}")
        elapsed += step_s

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        xs = [t for t, y in zip(ts, series["ttft_ms"]) if y is not None]
        ys = [y for y in series["ttft_ms"] if y is not None]
        ax1.plot(xs, ys, marker="o", color="tab:red")
        ax1.set_ylabel("TTFT (ms)")
        ax1.set_title(f"vLLM under load  {METRICS_URL}")
        ax1.grid(True, alpha=0.3)
        ax2.plot(ts, series["kv_pct"], marker="o", label="KV cache %")
        ax2.plot(ts, series["waiting"], marker="s", label="requests waiting")
        ax2.set_xlabel("seconds"); ax2.set_ylabel("KV % / waiting")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.tight_layout(); plt.show()
    return ts, series
