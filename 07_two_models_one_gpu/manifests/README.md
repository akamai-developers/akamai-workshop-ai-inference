# Two models on one GPU

`two-models.yaml` runs two vLLM servers, `vllm-fast` and `vllm-smart`, on a
single GPU. vLLM serves one model per process, so two models means two
Deployments. Both cap `--gpu-memory-utilization` at `0.45` so their weights and
KV caches fit one card together.

## GPU sharing is required

By default the NVIDIA device plugin gives each pod an exclusive GPU, so the
second server would stay `Pending`. To truly share one card, the cluster needs
**time-slicing** or **MPS** enabled. The workshop platform does this for you. On
your own cluster (Path B), enable time-slicing in the device plugin config, or
run the two servers on two cards instead.

## Apply it

```bash
kubectl apply -f two-models.yaml -n "$NAMESPACE"
kubectl rollout status deploy/vllm-fast -n "$NAMESPACE"
kubectl rollout status deploy/vllm-smart -n "$NAMESPACE"
```

Set `FAST_MODEL` and `SMART_MODEL` to the small models you want before applying.
The notebook points one client at `http://vllm-fast:8000/v1` and another at
`http://vllm-smart:8000/v1` and routes between them.

## Routing options

The module uses a simple Python router that picks an endpoint based on the request.
In production you would put a gateway in front so clients send one `model` field
and the gateway routes. [agentgateway](https://agentgateway.dev/) is the open
source option for that. The lab keeps the router in Python so the routing logic
is visible and you do not need to deploy a gateway to see contention.
