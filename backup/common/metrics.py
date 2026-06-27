"""Scrape and parse the vLLM ``/metrics`` endpoint.

vLLM exposes Prometheus metrics in the standard text exposition format. You do
not need a Prometheus server to read them. This module fetches the raw text,
parses the gauges and counters you care about, and gives you a tiny recorder so
you can sample a value over time and plot it.

The metric names used across the labs:

- ``vllm:num_requests_running``     requests in the running batch right now
- ``vllm:num_requests_waiting``     requests queued, waiting for a batch slot
- ``vllm:kv_cache_usage_perc``      fraction of the KV cache blocks in use (0..1).
  vLLM's V1 engine renamed this from ``vllm:gpu_cache_usage_perc``; snapshot()
  reads whichever the server exposes and returns it under BOTH names.
- ``vllm:num_preemptions_total``    cumulative requests evicted under pressure
- ``vllm:time_to_first_token_seconds``  TTFT histogram (prefill latency)
- ``vllm:time_per_output_token_seconds`` inter-token latency (decode)
"""

import time
from dataclasses import dataclass, field

import requests


# The gauges and counters the labs read by name. Histograms (TTFT, TPOT) are
# handled separately because they expose _sum / _count / _bucket series.
GAUGE_METRICS = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
]
COUNTER_METRICS = [
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]


def fetch_metrics_text(metrics_url: str, timeout: float = 5.0) -> str:
    """Return the raw Prometheus text from the vLLM ``/metrics`` endpoint."""
    resp = requests.get(metrics_url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_metrics(text: str) -> dict:
    """Parse the exposition text into ``{metric_name: value}``.

    We sum the values across label sets for each metric name, which is what you
    want for a single-model server: one number per metric. Lines starting with
    ``#`` (HELP and TYPE) and histogram bucket series are skipped here; use
    :func:`parse_histogram_avg` for histograms.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # A sample line looks like: name{label="x"} 1.23   (the name may have no labels)
        name_part, _, value_part = line.partition(" ")
        if not value_part:
            continue
        base_name = name_part.split("{", 1)[0]
        try:
            value = float(value_part.strip())
        except ValueError:
            continue
        values[base_name] = values.get(base_name, 0.0) + value
    return values


def parse_histogram_avg(text: str, metric: str) -> float | None:
    """Return the running average for a Prometheus histogram metric.

    Histograms expose ``<metric>_sum`` and ``<metric>_count``. The average since
    server start is ``sum / count``. For TTFT that is the average prefill
    latency; for TPOT it is the average decode latency per token.
    """
    parsed = parse_metrics(text)
    total = parsed.get(f"{metric}_sum")
    count = parsed.get(f"{metric}_count")
    if total is None or not count:
        return None
    return total / count


def snapshot(metrics_url: str) -> dict:
    """Fetch and parse the gauges and counters the labs care about.

    vLLM's V1 engine renamed the KV-cache gauge from ``gpu_cache_usage_perc`` to
    ``kv_cache_usage_perc``. We return the real value under BOTH names so a lab
    reads correct KV usage no matter which vLLM version your server runs.
    """
    text = fetch_metrics_text(metrics_url)
    parsed = parse_metrics(text)
    wanted = GAUGE_METRICS + COUNTER_METRICS
    snap = {name: parsed.get(name, 0.0) for name in wanted}

    kv = parsed.get("vllm:kv_cache_usage_perc")
    if kv is None:
        kv = parsed.get("vllm:gpu_cache_usage_perc", 0.0)
    snap["vllm:kv_cache_usage_perc"] = kv
    snap["vllm:gpu_cache_usage_perc"] = kv
    return snap


@dataclass
class SeriesRecorder:
    """Sample one or more metrics over time so you can plot them.

    Usage::

        rec = SeriesRecorder(settings.metrics_url)
        for _ in range(30):
            rec.record()
            time.sleep(1)
        rec.plot("vllm:num_requests_running")
    """

    metrics_url: str
    timestamps: list[float] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    _start: float | None = None

    def record(self) -> dict:
        """Take one sample and store it with a relative timestamp."""
        now = time.monotonic()
        if self._start is None:
            self._start = now
        sample = snapshot(self.metrics_url)
        self.timestamps.append(now - self._start)
        self.samples.append(sample)
        return sample

    def series(self, metric: str) -> list[float]:
        """Return the recorded values for one metric as a list."""
        return [s.get(metric, 0.0) for s in self.samples]

    def plot(self, *metrics: str, title: str | None = None):
        """Plot one or more recorded metrics against elapsed seconds."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for metric in metrics:
            ax.plot(self.timestamps, self.series(metric), marker="o", label=metric)
        ax.set_xlabel("elapsed seconds")
        ax.set_ylabel("value")
        ax.set_title(title or "vLLM metrics over time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return ax
