"""
vllm_admin.py — switch the model your dedicated vLLM is serving, from the notebook.

Each student has their own vLLM Deployment named `vllm` in their own namespace, and a
scoped kubeconfig is mounted at ~/.kube/config whose current context already points at
that namespace. So the kubectl calls below need no -n flag and only touch the student's
own resources.

A vLLM server loads ONE model, fixed by --model at startup. "Switching" means patching
that flag and letting the Deployment's Recreate strategy restart the pod. Every model in
`AVAILABLE_MODELS` is pre-cached in the PVC, so the restart loads from cache in seconds,
not a re-download.

The 0.6B model is intentionally not listed here. It is a speculative decoding
draft model for Module 6, not a standalone target model students should serve.
"""
import json as _json
import subprocess

# Served target models pre-cached in the PVC (must match the deploy's --predownload-models).
AVAILABLE_MODELS = [
    "Qwen/Qwen3-4B",
    "RedHatAI/Qwen3-4B-FP8-dynamic",
]


def current_model() -> str:
    """Return the model the vLLM Deployment is currently configured to serve."""
    out = subprocess.run(
        ["kubectl", "get", "deployment", "vllm",
         "-o", "jsonpath={.spec.template.spec.containers[0].args[0]}"],
        check=True, capture_output=True, text=True,
    ).stdout
    return out.replace("--model=", "").strip()


def switch_model(model: str, timeout: str = "300s") -> str:
    """Restart this student's vLLM to serve `model`, loading from the PVC cache.

    Returns the model name once the new pod reports Ready. Set `model` in your
    OpenAI client requests afterward, or read it back from MODEL_NAME.
    """
    if model not in AVAILABLE_MODELS:
        allowed = ", ".join(AVAILABLE_MODELS)
        raise ValueError(
            f"{model!r} is not a supported served target model. "
            f"Allowed: {allowed}. The 0.6B model is only a speculative draft model."
        )
    patch = ('[{"op":"replace","path":"/spec/template/spec/containers/0/args/0",'
             f'"value":"--model={model}"}}]')
    subprocess.run(["kubectl", "patch", "deployment", "vllm",
                    "--type=json", "-p", patch], check=True)
    print(f"switching to {model} (Recreate restart, loads from the PVC cache)")
    subprocess.run(["kubectl", "rollout", "status",
                    "deployment/vllm", f"--timeout={timeout}"], check=True)
    print(f"now serving {model}. Use model=\"{model}\" in your client requests.")
    return model


def set_engine_arg(flag: str, value, timeout: str = "300s") -> list:
    """Set a vLLM engine flag and restart, for example:

        set_engine_arg("gpu-memory-utilization", 0.9)
        set_engine_arg("max-num-seqs", 256)

    It reads the current container args, replaces or appends `--flag=value`, patches the
    Deployment, and waits for the rollout. The pod reloads the model from the PVC cache.
    Returns the new args list.
    """
    raw = subprocess.run(
        ["kubectl", "get", "deployment", "vllm",
         "-o", "jsonpath={.spec.template.spec.containers[0].args}"],
        check=True, capture_output=True, text=True,
    ).stdout
    args = _json.loads(raw)
    token = f"--{flag}={value}"
    for i, a in enumerate(args):
        if a.startswith(f"--{flag}="):
            args[i] = token
            break
    else:
        args.append(token)
    patch = _json.dumps([{"op": "replace",
                          "path": "/spec/template/spec/containers/0/args",
                          "value": args}])
    subprocess.run(["kubectl", "patch", "deployment", "vllm", "--type=json", "-p", patch], check=True)
    print(f"set --{flag}={value}, restarting from the PVC cache")
    subprocess.run(["kubectl", "rollout", "status", "deployment/vllm", f"--timeout={timeout}"], check=True)
    print(f"vLLM restarted with --{flag}={value}")
    return args


# The raw kubectl the helper runs, for the kubectl-teaching labs:
#   kubectl patch deployment vllm --type=json \
#     -p='[{"op":"replace","path":"/spec/template/spec/containers/0/args/0","value":"--model=RedHatAI/Qwen3-4B-FP8-dynamic"}]'
#   kubectl rollout status deployment/vllm
