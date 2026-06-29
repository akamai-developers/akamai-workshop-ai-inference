"""Foundation-module helpers that the shared common/ package does not expose.

The optimization modules (5-8) import the shared common/ files, so this build does
NOT edit them. The foundation notebooks need three things the source notebooks had
that common/ does not provide directly:

- ``kv_cache_info``: read the KV-cache pool shape (block size, number of GPU blocks,
  derived token capacity) from the server. vLLM publishes these as labels on the
  Prometheus ``vllm:cache_config_info`` metric, which common/metrics.py does not
  parse (it sums numeric samples and skips info-style label sets).
- runtime metric-name resolution: vLLM has renamed gauges across versions, so the
  notebooks resolve the real name the server emits rather than hardcoding one.
- a small streaming helper that times the first token and the decode rate.

Everything reads the live server. With no endpoint reachable the functions raise a
clear connection error, which is the documented behavior off cluster.
"""

import time
from dataclasses import dataclass

import requests

from .config import get_settings


# Friendly name -> ordered list of real metric names to try. First present wins.
# vLLM's V1 engine renamed gpu_cache_usage_perc to kv_cache_usage_perc; accept both.
_ALIASES = {
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "requests_running": ["vllm:num_requests_running"],
    "requests_waiting": ["vllm:num_requests_waiting"],
    "preemptions": ["vllm:num_preemptions_total"],
    "prefix_cache_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "prefix_cache_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "generation_tokens": ["vllm:generation_tokens_total"],
    "ttft": ["vllm:time_to_first_token_seconds"],
    "e2e_latency": ["vllm:e2e_request_latency_seconds"],
}


@dataclass
class _Sample:
    labels: dict
    value: float


class MetricsText:
    """A parsed /metrics snapshot with alias resolution and info-label lookup.

    The Prometheus text format is line oriented::

        metric_name{label="v",...} 12.0

    Histograms expose ``_bucket`` / ``_sum`` / ``_count`` series. Info metrics
    (such as ``vllm:cache_config_info``) carry all their data in the labels with a
    value of 1.0, which is why a plain numeric parser cannot read them.
    """

    def __init__(self, text: str):
        self._samples: dict[str, list[_Sample]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, labels, value = self._parse_line(line)
            if name is not None:
                self._samples.setdefault(name, []).append(_Sample(labels, value))

    @staticmethod
    def _parse_line(line: str):
        try:
            if "{" in line:
                name = line[: line.index("{")]
                labelstr = line[line.index("{") + 1: line.rindex("}")]
                rest = line[line.rindex("}") + 1:].strip()
                labels = {}
                for pair in _split_labels(labelstr):
                    if "=" in pair:
                        k, _, v = pair.partition("=")
                        labels[k.strip()] = v.strip().strip('"')
                return name, labels, float(rest.split()[0])
            parts = line.split()
            return parts[0], {}, float(parts[1])
        except (ValueError, IndexError):
            return None, None, None

    def resolve(self, friendly: str):
        """Real metric name the server emits for a friendly key, or None."""
        for cand in _ALIASES.get(friendly, [friendly]):
            if cand in self._samples or (cand + "_count") in self._samples:
                return cand
        return None

    def gauge(self, friendly: str):
        """Single value for a gauge/counter, summed across replicas, or None."""
        real = self.resolve(friendly)
        if real is None:
            return None
        vals = [s.value for s in self._samples.get(real, [])]
        return sum(vals) if vals else None

    def histogram_avg(self, friendly: str):
        """Average of a histogram = sum / count, by friendly name. None if empty.

        vLLM reports latencies (TTFT, end-to-end) as histograms that expose
        ``<name>_sum`` and ``<name>_count``. The average since server start is the
        ratio. The base name is resolved at runtime so version renames are handled.
        """
        real = self.resolve(friendly)
        if real is None:
            return None
        total = sum(s.value for s in self._samples.get(real + "_sum", []))
        count = sum(s.value for s in self._samples.get(real + "_count", []))
        return (total / count) if count else None

    def info(self, name: str) -> dict:
        """Labels of an _info metric (value carries no data, labels do)."""
        s = self._samples.get(name)
        return dict(s[0].labels) if s else {}


def _split_labels(labelstr: str) -> list:
    out, buf, in_quote = [], [], False
    for ch in labelstr:
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
        elif ch == "," and not in_quote:
            out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def fetch_metrics(settings=None) -> MetricsText:
    """GET the server's /metrics and parse it. Raises on an unreachable endpoint."""
    settings = settings or get_settings()
    r = requests.get(
        settings.metrics_url,
        headers={"Authorization": f"Bearer {settings.api_key}"},
        timeout=15.0,
    )
    r.raise_for_status()
    return MetricsText(r.text)


def resolve_metric_name(friendly: str, settings=None):
    """Return the real vLLM metric name the running server emits, or None.

    Use this instead of hardcoding a gauge name. vLLM renames metrics across
    versions, so the notebooks ask the server what it actually exposes.
    """
    return fetch_metrics(settings).resolve(friendly)


def read_gauge(friendly: str, settings=None):
    """One gauge value by friendly name, resolved at runtime. None if absent."""
    return fetch_metrics(settings).gauge(friendly)


def served_model_name(settings=None) -> str:
    """The model id the running server actually serves, or the configured name.

    Asks the server's ``/v1/models`` and returns the first id, so the notebooks use the
    model that is really deployed rather than a configured default. Falls back to
    ``get_settings().model_name`` (the ``MODEL_NAME`` value) when the server cannot be
    reached, so off-cluster cells still resolve a usable name.
    """
    settings = settings or get_settings()
    try:
        root = settings.vllm_host.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        r = requests.get(
            f"{root}/v1/models",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=10.0,
        )
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        return ids[0] if ids else settings.model_name
    except Exception:
        return settings.model_name


def kv_cache_info(settings=None) -> dict:
    """Read the KV-cache pool shape and live usage from the server.

    Returns ``block_size``, ``num_gpu_blocks``, the derived ``capacity_tokens``
    (block_size x num_gpu_blocks), ``gpu_memory_utilization``, whether prefix
    caching is on, the cache dtype, and current usage fraction. The pool fields
    come from the ``vllm:cache_config_info`` labels.
    """
    m = fetch_metrics(settings)
    info = m.info("vllm:cache_config_info")

    def _int(key):
        try:
            return int(info[key])
        except (KeyError, ValueError, TypeError):
            return None

    block_size = _int("block_size")
    num_gpu_blocks = _int("num_gpu_blocks")
    capacity = (block_size * num_gpu_blocks
                if block_size is not None and num_gpu_blocks is not None else None)
    return {
        "block_size": block_size,
        "num_gpu_blocks": num_gpu_blocks,
        "capacity_tokens": capacity,
        "gpu_memory_utilization": info.get("gpu_memory_utilization"),
        "enable_prefix_caching": info.get("enable_prefix_caching"),
        "cache_dtype": info.get("cache_dtype"),
        "kv_cache_usage_perc": m.gauge("kv_cache_usage"),
    }


def load_model_config(repo: str, fallback_dir=None) -> dict:
    """Return a model's ``config.json`` as a dict.

    Reads the real config from the Hugging Face Hub. If the Hub is unreachable
    (a notebook pod with no egress, for example), it falls back to a bundled copy
    at ``<fallback_dir>/<model>.config.json`` so the inspection still uses the
    real architecture numbers. ``<model>`` is the part of ``repo`` after the slash.
    """
    try:
        import json
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(repo, "config.json")))
    except Exception:
        if fallback_dir is None:
            raise
        import json, os
        name = repo.split("/")[-1]
        with open(os.path.join(fallback_dir, f"{name}.config.json")) as f:
            return json.load(f)


def model_param_count(repo: str, fallback_dir=None) -> dict:
    """Return ``{dtype: count}`` of a model's parameters from safetensors metadata.

    Reads the real safetensors headers from the Hub (no weights are downloaded).
    Falls back to a bundled ``<fallback_dir>/params.json`` mapping ``repo`` to a
    ``{dtype: count}`` dict when the Hub is unreachable.
    """
    try:
        from huggingface_hub import get_safetensors_metadata
        return dict(get_safetensors_metadata(repo).parameter_count)
    except Exception:
        if fallback_dir is None:
            raise
        import json, os
        with open(os.path.join(fallback_dir, "params.json")) as f:
            return json.load(f)[repo]


def stream_and_time(client, model, messages, max_tokens=128, temperature=0.0, extra=None):
    """Stream one chat request and return TTFT, decode rate, and the text.

    Returns a dict with ``ttft_s`` (time to first token, the prefill latency seen
    from the client), ``tpot_s`` (mean time per output token during decode),
    ``tokens`` (number of streamed content chunks), and ``text``. Timing is
    client side, so it includes the network hop; the server's own histograms
    (read through common/metrics or this module) are the ground truth.
    """
    extra = extra or {}
    start = time.time()
    first = None
    stamps = []
    parts = []
    stream = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, stream=True, **extra,
    )
    for chunk in stream:
        if not chunk.choices or not chunk.choices[0].delta.content:
            continue
        now = time.time()
        if first is None:
            first = now
        stamps.append(now)
        parts.append(chunk.choices[0].delta.content)
    ttft = (first - start) if first else None
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    tpot = (sum(gaps) / len(gaps)) if gaps else None
    return {"ttft_s": ttft, "tpot_s": tpot, "tokens": len(stamps), "text": "".join(parts)}


def sa_agent_loop(steps=4, obs_tokens=200, max_tokens=120, client=None, model=None, settings=None):
    """Run the Akamai Solutions Architect agent as a ReAct loop, one record per step.

    Same mechanic as ``common/agent_loop.step``: each step re-sends a growing prompt, so
    ``prompt_tokens`` climbs and you watch the context tax. The persona and the task are the
    Module 9 agent's, an Akamai solutions architect pricing GPUs, so the load you measure here
    is the load you deploy at the end of the workshop. Thinking is off for clean measurement.
    """
    from .config import build_client
    from . import agent_loop
    settings = settings or get_settings()
    client = client or build_client(settings)
    model = model or settings.model_name
    system = (
        "You are the Akamai Cloud Solutions Architect agent. Work in a ReAct loop: write a "
        "Thought, then an Action (one tool call), then read the Observation, then repeat until "
        "you can answer. Your tool is akamai_gpu_pricing(card), which returns the per-hour and "
        "per-month price of an Akamai Cloud GPU. Keep thoughts short and cite the prices you read."
    )
    filler = ("Pricing row: RTX 4000 Ada x1 is $0.52/hr ($374.40/mo); RTX PRO 6000 Blackwell x1 "
              "is $2.50/hr; both bill hourly and the account starts with free credit. ")

    def observation(i):
        chars = max(8, int(obs_tokens) * 4)
        body = (filler * (chars // len(filler) + 1))[:chars]
        return f"Observation {i}: akamai_gpu_pricing returned a row.\n{body}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Recommend the cheapest Akamai GPU to serve a 4B model, and when to size up."},
    ]
    records = []
    for i in range(1, steps + 1):
        r = agent_loop.step(client, model, messages, max_tokens=max_tokens)
        r["step"] = i
        records.append(r)
        if not r["ok"]:
            break
        messages.append({"role": "assistant", "content": r["content"]})
        messages.append({"role": "user", "content": observation(i)})
    return records
