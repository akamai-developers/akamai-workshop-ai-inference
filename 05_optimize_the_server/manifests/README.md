# Baseline vLLM manifest

`vllm-baseline.yaml` is the deployment you tune in Module 5. It is under-tuned
on purpose. Your job in the module is to change the four engine flags, redeploy, and
prove the gain by re-running the load sweep.

## The four flags you change

| Flag | Baseline | What it does | The tradeoff |
|---|---|---|---|
| `--gpu-memory-utilization` | `0.40` | Fraction of GPU memory vLLM may use. Most of it becomes KV cache. | Higher means more cache and more concurrency, until you risk an out-of-memory at load. Often the biggest single win. |
| `--max-model-len` | `2048` | Longest context (prompt + output) the engine reserves cache for. | Higher allows longer requests but each running request reserves more KV blocks, so fewer fit at once. |
| `--max-num-seqs` | `16` | Max requests in the running batch. | Higher raises throughput until the KV cache runs out; too high causes preemption. |
| `--max-num-batched-tokens` | `2048` | Max tokens the scheduler batches per step. | Higher lets larger prefills run together (better throughput) at the cost of longer per-step latency. |

## Why a restart is needed

These are engine initialization arguments. vLLM reads them once at startup to size
the KV cache and the scheduler. They are not runtime-adjustable, so changing them
means restarting the pod. In Kubernetes that is `kubectl rollout restart`.

## Apply it

```bash
kubectl apply -f vllm-baseline.yaml -n "$NAMESPACE"
kubectl rollout status deploy/vllm -n "$NAMESPACE"
```

The Service is named `vllm` on port 8000, matching the default
`VLLM_HOST=http://vllm:8000/v1` the notebooks expect. Set `MODEL_NAME` to the
model your environment serves before you apply.
