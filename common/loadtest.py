"""
loadtest.py — stress your own vLLM with concurrency and token volume.

One uniform tool for every load lab. It fires chat requests at the student's own vLLM
(VLLM_HOST, no auth, allowed by the NetworkPolicy), with two knobs:

  - concurrency:   how many requests are in flight at once (drives batching and the queue)
  - input_tokens / output_tokens:  prompt and answer size (drives prefill, the KV cache, decode)

Each request gets a unique prefix, so prefix caching does not hide the KV cache cost.

Three ways to use it:

  # 1. one-shot summary at a single concurrency level
  loadtest.run(concurrency=32, input_tokens=512, output_tokens=128)

  # 2. a sweep, one summary row per level (the saturation walk)
  loadtest.sweep([1, 8, 32, 64, 128])

  # 3. drive load in the background while you watch the metrics climb
  from common import ttft_metrics
  with loadtest.background(concurrency=32, input_tokens=512):
      ttft_metrics.watch(30)
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import get_settings, build_client, chat_extra

_FILLER = (
    "GPU memory bandwidth is the bottleneck for transformer inference, because every "
    "decode step reads the whole weight set and the per request KV cache. Continuous "
    "batching amortizes the weight read across requests when there is cache headroom. "
)


def _prompt(input_tokens: int, tag: str) -> str:
    """A prompt of about `input_tokens` tokens (4 chars each), unique per `tag`."""
    chars = max(8, int(input_tokens) * 4)
    reps = chars // len(_FILLER) + 1
    body = (_FILLER * reps)[:chars]
    return f"Request {tag}. {body}\n\nSummarize the text above in one sentence."


def _one(client, model, input_tokens, output_tokens, extra, tag):
    """Stream one request. Return time to first token, total time, and token count."""
    start = time.time()
    first = None
    n = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _prompt(input_tokens, tag)}],
            max_tokens=output_tokens, temperature=0.0, stream=True, **extra,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.time()
                n += 1
        end = time.time()
        return {"ok": True, "ttft": (first - start) if first else None,
                "total": end - start, "tokens": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _pct(values, q):
    s = sorted(v for v in values if v is not None)
    return s[min(int(q * len(s)), len(s) - 1)] if s else None


def run(concurrency=16, num_requests=None, input_tokens=256, output_tokens=128,
        model=None, settings=None):
    """Fire `num_requests` at the given concurrency and return aggregate stats."""
    settings = settings or get_settings()
    client = build_client(settings)
    model = model or settings.model_name
    extra = chat_extra(model)
    num_requests = num_requests or max(concurrency * 4, 8)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        res = list(pool.map(
            lambda i: _one(client, model, input_tokens, output_tokens, extra, str(i)),
            range(num_requests),
        ))
    wall = time.time() - t0

    ok = [r for r in res if r.get("ok")]
    tokens = sum(r["tokens"] for r in ok)
    ttfts = [r["ttft"] for r in ok]
    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "ok": len(ok),
        "failed": len(res) - len(ok),
        "throughput_tok_s": round(tokens / wall, 1) if wall else 0.0,
        "ttft_p50_ms": round(_pct(ttfts, 0.50) * 1000) if _pct(ttfts, 0.50) else None,
        "ttft_p95_ms": round(_pct(ttfts, 0.95) * 1000) if _pct(ttfts, 0.95) else None,
        "wall_s": round(wall, 1),
    }


def sweep(levels=(1, 8, 32, 64), input_tokens=256, output_tokens=128, model=None):
    """Run `run()` at each concurrency level and print one row each. Returns the rows."""
    rows = []
    for c in levels:
        r = run(concurrency=c, input_tokens=input_tokens, output_tokens=output_tokens, model=model)
        rows.append(r)
        print(f"c={c:>3}  {r['throughput_tok_s']:>8} tok/s  "
              f"TTFT p50 {r['ttft_p50_ms']} ms  p95 {r['ttft_p95_ms']} ms  "
              f"ok {r['ok']}/{r['requests']}")
    return rows


class _Background:
    """Keep firing requests at a fixed concurrency until the block exits."""

    def __init__(self, concurrency, input_tokens, output_tokens, model, settings):
        self.concurrency = concurrency
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.settings = settings or get_settings()
        self.client = build_client(self.settings)
        self.model = model or self.settings.model_name
        self.extra = chat_extra(self.model)
        self._stop = threading.Event()
        self._threads = []
        self._n = 0

    def _worker(self, wid):
        i = 0
        while not self._stop.is_set():
            _one(self.client, self.model, self.input_tokens, self.output_tokens,
                 self.extra, f"{wid}-{i}")
            i += 1

    def start(self):
        for w in range(self.concurrency):
            t = threading.Thread(target=self._worker, args=(w,), daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def stop(self):
        self._stop.set()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def background(concurrency=32, input_tokens=256, output_tokens=128, model=None, settings=None):
    """Drive continuous load in the background. Use as a context manager with watch()."""
    return _Background(concurrency, input_tokens, output_tokens, model, settings)
