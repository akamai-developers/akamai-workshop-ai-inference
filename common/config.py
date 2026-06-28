"""Shared configuration for the workshop labs.

Every notebook reads its connection details from environment variables so the
same code runs in both prerequisite paths:

- Path A (hosted workshop): the values are already set in your JupyterLab
  environment. You do not touch them.
- Path B (bring your own): you export them yourself before launching Jupyter,
  or drop them in a .env file that your shell sources.

Nothing here is hardcoded to a specific cluster. The defaults point at an
in-cluster service name (``vllm``) that resolves only inside Kubernetes, so off
cluster a missing variable surfaces as a clear connection error to that name
rather than silently reaching someone else's endpoint.
"""

import os
import subprocess
from dataclasses import dataclass


# Default base URL assumes the vLLM Service is reachable as ``vllm`` on port
# 8000 inside your namespace. Override with VLLM_HOST when you run elsewhere.
DEFAULT_VLLM_HOST = "http://vllm:8000/v1"
DEFAULT_MODEL_NAME = "RedHatAI/Qwen3-4B-FP8-dynamic"
DEFAULT_NAMESPACE = "default"


@dataclass
class Settings:
    """Resolved connection details for the current environment."""

    vllm_host: str
    model_name: str
    namespace: str
    kubeconfig: str
    api_key: str

    @property
    def metrics_url(self) -> str:
        """The vLLM Prometheus endpoint, derived from the OpenAI base URL.

        vLLM serves ``/metrics`` at the server root, not under ``/v1``. We strip
        a trailing ``/v1`` so the two URLs stay in sync from one variable.
        """
        root = self.vllm_host.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root}/metrics"


def _resolve_namespace() -> str:
    """Best-effort current Kubernetes namespace.

    Precedence: the NAMESPACE env var if set; else the namespace of the mounted
    kubeconfig's current context (the hosted platform scopes that context to your
    own namespace, e.g. ``workshop-s01``); else ``default``. The hosted workspace
    pod mounts no service-account token, so we ask kubectl rather than reading
    ``/var/run/secrets``. This keeps ``kubectl -n {namespace}`` in the tuning and
    agent labs pointed at the right namespace even when NAMESPACE is
    not injected as an env var.
    """
    ns = os.environ.get("NAMESPACE")
    if ns:
        return ns
    try:
        out = subprocess.run(
            ["kubectl", "config", "view", "--minify", "-o", "jsonpath={..namespace}"],
            capture_output=True, text=True, timeout=5,
        )
        ns = out.stdout.strip()
        if ns:
            return ns
    except Exception:
        pass
    return DEFAULT_NAMESPACE


def get_settings() -> Settings:
    """Read settings from the environment, falling back to sane defaults."""
    return Settings(
        vllm_host=os.environ.get("VLLM_HOST", DEFAULT_VLLM_HOST),
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        namespace=_resolve_namespace(),
        kubeconfig=os.environ.get("KUBECONFIG", ""),
        # vLLM does not require a key by default. The platform sets a placeholder
        # so the OpenAI client, which insists on a non-empty key, stays happy.
        api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
    )


def chat_extra(model_name: str) -> dict:
    """Per-model kwargs for chat.completions.create.

    The workshop serves the hybrid Qwen3-4B, which thinks (emits a ``<think>`` block)
    by default. The measurement labs turn that off (a Qwen3-only chat-template flag) so
    the completions you read and the tokens you benchmark are the answer, not the model
    thinking out loud. Module 9's agent re-enables it (see ``agent/agent.py``) so it can
    reason while choosing tools. Returns an empty dict for non-Qwen3 models.
    """
    if "qwen3" in (model_name or "").lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def build_client(settings: Settings | None = None):
    """Return an OpenAI client pointed at your vLLM endpoint.

    The ``openai`` package is the same client you would use against the hosted
    OpenAI API. Pointing ``base_url`` at vLLM is the only change, which is the
    whole reason an OpenAI-compatible server is convenient.

    The returned client is lightly wrapped so calls behave against vLLM: Qwen3
    reasoning is turned off (see ``chat_extra``), and an empty ``tools=[]`` is dropped
    because vLLM 0.20+ rejects it. Both are no-ops for other models and for calls that
    pass real tools, so notebooks call ``create`` normally. The wrapper lives on this
    client instance, so copying it with ``client.with_options(...)`` drops it; that is
    why ``common/load.py`` re-applies ``chat_extra`` at the call site rather than
    relying on the wrapper surviving a copy.
    """
    from openai import OpenAI

    settings = settings or get_settings()
    client = OpenAI(base_url=settings.vllm_host, api_key=settings.api_key)

    extra = chat_extra(settings.model_name)
    _create = client.chat.completions.create

    def create(*args, **kwargs):
        if extra:
            eb = kwargs.setdefault("extra_body", {})
            eb.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
        if "tools" in kwargs and not kwargs["tools"]:
            kwargs.pop("tools")
            kwargs.pop("tool_choice", None)
        return _create(*args, **kwargs)

    client.chat.completions.create = create
    return client


def print_settings(settings: Settings | None = None) -> Settings:
    """Print the resolved values so you can see your own environment.

    Call this at the top of every lab. It never prints the API key value, only
    whether one is set, so screenshots stay safe to share.
    """
    settings = settings or get_settings()
    print("Resolved workshop settings")
    print("-" * 40)
    print(f"  VLLM_HOST   : {settings.vllm_host}")
    print(f"  metrics URL : {settings.metrics_url}")
    print(f"  MODEL_NAME  : {settings.model_name}")
    print(f"  NAMESPACE   : {settings.namespace}")
    print(f"  KUBECONFIG  : {settings.kubeconfig or '(using in-cluster or default context)'}")
    print(f"  API key set : {'yes' if settings.api_key and settings.api_key != 'not-needed' else 'no (vLLM open by default)'}")
    return settings
