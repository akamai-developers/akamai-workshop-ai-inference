"""
agent_loop.py - a tiny ReAct agent you run against your own vLLM and watch on /metrics.

A chatbot is one request: one prefill, one decode. An agent is a LOOP. Every step it re-sends a
growing prompt (the system prompt, the tool schemas, and every Thought, Action, and Observation so
far), decodes a reasoning trace, calls a tool, and goes again. This helper IS that loop, small
enough to read, so every module can point it at one mechanic: context growth, prefix caching,
fan-out, or tail latency.

No real tools are needed. A fake tool returns a fixed-size observation, so you control how fast the
context grows. Everything reads VLLM_HOST through the workshop's build_client (thinking off for
clean measurement) and times the first token client-side by streaming.
"""
import time

from .config import get_settings, build_client, chat_extra


def build_agent_prompt(n_tools: int = 20) -> tuple:
    """A realistic agent system prompt: a ReAct instruction plus n fake tool schemas.

    Each schema is roughly 40 to 60 tokens, so n_tools sets the prefix size. Returns the prompt and
    a rough token estimate (4 chars per token). Keep the estimate under about 1800 so a turn plus a
    short answer still fits the 2048 cap.
    """
    tools = "\n".join(
        f"- tool_{i}(query: str, top_k: int = 5, filters: dict = {{}}) -> json"
        f"  // searches corpus {i} and returns the top_k ranked passages with scores"
        for i in range(n_tools)
    )
    system = (
        "You are an autonomous research agent. Work in a ReAct loop: write a Thought, then an "
        "Action (one tool call), then read the Observation, and repeat until you can answer.\n"
        "Rules: cite sources, never fabricate, one tool call per step, keep thoughts short.\n\n"
        "Available tools:\n" + tools
    )
    return system, len(system) // 4


def observation(step_num: int, approx_tokens: int = 300) -> str:
    """A fake tool result of about approx_tokens tokens, so you control the context-growth rate."""
    filler = ("Passage: GPU memory bandwidth bounds decode, because each token reloads the weights "
              "and the per-request KV cache from VRAM. ")
    chars = max(8, int(approx_tokens) * 4)
    body = (filler * (chars // len(filler) + 1))[:chars]
    return f"Observation {step_num}: the tool returned 3 ranked hits.\n{body}"


def step(client, model, messages, max_tokens: int = 200) -> dict:
    """Run one agent step: stream the model, time the first token, and read token usage.

    Returns ok, ttft_ms, prompt_tokens, completion_tokens, and content. On a server error (for
    example exceeding the context cap) returns ok=False with the error string.
    """
    extra = chat_extra(model)
    t0 = time.perf_counter()
    first = None
    parts = []
    prompt_tokens = completion_tokens = 0
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
            stream=True, stream_options={"include_usage": True}, **extra,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter()
                parts.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens
        ttft = (first - t0) if first else (time.perf_counter() - t0)
        return {"ok": True, "ttft_ms": round(ttft * 1000, 1),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "content": "".join(parts)}
    except Exception as e:
        return {"ok": False, "error": str(e), "ttft_ms": None,
                "prompt_tokens": prompt_tokens, "completion_tokens": 0, "content": ""}


def run(client=None, model=None, steps: int = 5, n_tools: int = 20,
        obs_tokens: int = 300, max_tokens: int = 200, settings=None) -> list:
    """Run a ReAct loop and return one record per step.

    Each step appends the model's action and a fake observation to the running messages, so the
    prompt grows every turn. That growth is the agent context tax. The loop stops early if the
    server rejects a request for exceeding the context cap. Watch prompt_tokens climb and ttft_ms
    per step.
    """
    settings = settings or get_settings()
    client = client or build_client(settings)
    model = model or settings.model_name
    system, _approx = build_agent_prompt(n_tools)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Start: compare two GPUs for serving an agent."},
    ]
    records = []
    for i in range(1, steps + 1):
        r = step(client, model, messages, max_tokens=max_tokens)
        r["step"] = i
        records.append(r)
        if not r["ok"]:
            break
        # Append the model's action and the tool's observation: the prompt grows for next turn.
        messages.append({"role": "assistant", "content": r["content"]})
        messages.append({"role": "user", "content": observation(i, obs_tokens)})
    return records
