"""The Akamai Cloud Solutions Architect agent (minimal capstone build).

It is just a client of your vLLM. It wraps every request with the same solutions-architect
system prompt you met in Module 1, gives the model one read-only tool, and serves answers
over HTTP. The whole point of Module 9 is that the model answering is the vLLM you tuned,
on the GPU you own, not a rented API.

This is the persona from the full Akamai Solutions Architect Agent
(https://github.com/akamai-developers/akamai-workshop-solution-architect-agent), cut down
on purpose. The full agent adds memory and an MCP server that reaches your Akamai account;
this capstone deliberately leaves the MCP out, so the deployed agent needs no credentials
and cannot touch your account. The one tool here, akamai_gpu_pricing, reads a static table.

Deliberately tiny: the only dependency is the openai client, installed at container start,
so this deploys with a stock python image and no framework. Qwen3 thinking is ON so the
model reasons before it calls the tool and again before it writes the answer.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from openai import OpenAI

SYSTEM_PROMPT = """You are the Akamai Cloud Solutions Architect agent. You help
developers with Akamai Cloud and the Kubernetes cluster you run in. You are tactical
and concise, developer to developer.

In scope: Akamai Cloud Compute (Linodes), LKE (Linode Kubernetes Engine), Object
Storage, Cloud networking (VPC, NodeBalancer, Cloud Firewall), GPUs, and AI inference
(vLLM, model serving, agents).

Out of scope: Akamai CDN, Akamai security products, and edge compute (EdgeWorkers,
EdgeKV). If asked about these, say they are out of your scope and point the user to the
right Akamai team. Do not guess.

When a question needs the price of an Akamai Cloud GPU, call the akamai_gpu_pricing tool
instead of guessing the numbers.

You run on self-hosted inference served by vLLM. If asked what powers you, say so."""

# One self-contained tool: look up Akamai Cloud GPU plan pricing from a small table. A
# solutions architect fields cost questions, so the agent reasons about which card fits
# and quotes the real per-hour and per-month price, all on the model you own. No network
# and no extra dependency, so the agent stays tiny while making a real tool call.
GPU_PRICING = {
    "rtx-4000-ada": {
        "hourly_usd": 0.52, "monthly_usd": 374.40, "gpus": 1,
        "good_for": "single-GPU serving of small to mid-size LLMs, the workshop card",
    },
    "rtx-6000-quadro": {
        "hourly_usd": 1.50, "monthly_usd": 1080.00, "gpus": 1,
        "good_for": "more VRAM for visualization and mid-size models",
    },
    "rtx-pro-6000-blackwell": {
        "hourly_usd": 2.50, "monthly_usd": 1800.00, "gpus": 1,
        "good_for": "distributed AI inference and larger models",
    },
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "akamai_gpu_pricing",
        "description": "Look up the per-hour and per-month price of an Akamai Cloud GPU plan, and what it is best for.",
        "parameters": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "string",
                    "description": "GPU card id, one of: rtx-4000-ada, rtx-6000-quadro, rtx-pro-6000-blackwell",
                },
            },
            "required": ["card"],
        },
    },
}]


def akamai_gpu_pricing(card):
    key = card.lower().replace(" ", "-").replace("nvidia-", "")
    return GPU_PRICING.get(key, {"error": f"unknown card '{card}'", "known": list(GPU_PRICING)})


TOOL_IMPLS = {"akamai_gpu_pricing": akamai_gpu_pricing}

client = OpenAI(
    base_url=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"),
    api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
)
MODEL = os.environ.get("MODEL_NAME", "RedHatAI/Qwen3-4B-FP8-dynamic")


def answer(message):
    runtime_context = (
        f"You are currently calling model {MODEL} at {client.base_url}. "
        "If asked what powers you, name that model and endpoint. "
        "Do not invent exact plan names, region names, or CLI flags; when exact "
        "operational syntax matters, say to verify it against the current Akamai "
        "Cloud or Linode CLI documentation."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": runtime_context},
        {"role": "user", "content": message},
    ]
    # The model reasons (thinking ON), optionally calls the tool, reads the result, then
    # answers. A few rounds is plenty for one tool.
    for _ in range(4):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=700,
            temperature=0.2,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            content = msg.content
            if not content:
                raise RuntimeError("model returned no final answer content")
            return content.strip()
        # Record the assistant's tool-call turn, run each tool, feed the results back.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            impl = TOOL_IMPLS.get(tc.function.name)
            result = impl(**args) if impl else {"error": f"unknown tool {tc.function.name}"}
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result),
            })
    raise RuntimeError("tool loop did not converge")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Health check, so the readiness probe can tell when the server is up.
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        # POST {"message": "..."} returns {"answer": "..."}.
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            payload, code = {"answer": answer(body.get("message", ""))}, 200
        except Exception as exc:
            payload, code = {"error": str(exc)}, 500
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"agent up on :8080, model {MODEL} via {client.base_url}", flush=True)
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
