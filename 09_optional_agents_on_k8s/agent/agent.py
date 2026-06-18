"""A small Akamai Cloud Solutions Architect agent.

It is just a client of your vLLM. It wraps every request with a solutions-architect
system prompt and serves answers over HTTP. The whole point of Module 9 is that the
model answering is the vLLM you tuned, on the GPU you own, not a rented API.

Deliberately tiny: the only dependency is the openai client, installed at container
start, so this deploys with a stock python image and no framework. The persona is the
same one the full Akamai Solutions Architect Agent workshop builds, kept to chat only
here so the capstone stays about inference, not agent plumbing.
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

You run on self-hosted inference served by vLLM. If asked what powers you, say so."""

client = OpenAI(
    base_url=os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1"),
    api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
)
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")

# Unlike the measurement labs, the agent leaves Qwen3 thinking ON so it can reason
# before it picks a tool. The server's --reasoning-parser splits that <think> block
# into reasoning_content, so the answer we return below stays clean. We still drop an
# empty tools=[], which vLLM 0.20+ rejects, so adding a tool later is a one-line change.
_create = client.chat.completions.create


def _vllm_create(*args, **kwargs):
    if "qwen3" in MODEL.lower():
        kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", True)
    if "tools" in kwargs and not kwargs["tools"]:
        kwargs.pop("tools")
        kwargs.pop("tool_choice", None)
    return _create(*args, **kwargs)


client.chat.completions.create = _vllm_create


def answer(message):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


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
