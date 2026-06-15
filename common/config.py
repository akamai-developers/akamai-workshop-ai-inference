"""Shared configuration for the workshop labs.

Every notebook reads its connection details from environment variables so the
same code runs in both prerequisite paths:

- Path A (hosted workshop): the values are already set in your JupyterLab
  environment. You do not touch them.
- Path B (bring your own): you export them yourself before launching Jupyter,
  or drop them in a .env file that your shell sources.

Nothing here is hardcoded to a specific cluster. The defaults point at an
in-cluster service name (``vllm``) that resolves only inside Kubernetes, so a
missing variable fails loudly instead of leaking someone else's endpoint.
"""

import os
from dataclasses import dataclass


# Default base URL assumes the vLLM Service is reachable as ``vllm`` on port
# 8000 inside your namespace. Override with VLLM_HOST when you run elsewhere.
DEFAULT_VLLM_HOST = "http://vllm:8000/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
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


def get_settings() -> Settings:
    """Read settings from the environment, falling back to sane defaults."""
    return Settings(
        vllm_host=os.environ.get("VLLM_HOST", DEFAULT_VLLM_HOST),
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        namespace=os.environ.get("NAMESPACE", DEFAULT_NAMESPACE),
        kubeconfig=os.environ.get("KUBECONFIG", ""),
        # vLLM does not require a key by default. The platform sets a placeholder
        # so the OpenAI client, which insists on a non-empty key, stays happy.
        api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
    )


def build_client(settings: Settings | None = None):
    """Return an OpenAI client pointed at your vLLM endpoint.

    The ``openai`` package is the same client you would use against the hosted
    OpenAI API. Pointing ``base_url`` at vLLM is the only change, which is the
    whole reason an OpenAI-compatible server is convenient.
    """
    from openai import OpenAI

    settings = settings or get_settings()
    return OpenAI(base_url=settings.vllm_host, api_key=settings.api_key)


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
