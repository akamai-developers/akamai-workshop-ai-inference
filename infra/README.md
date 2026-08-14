# infra/: the Path B paved path

Terraform that creates the cluster this workshop's manifests expect: one LKE
cluster with a CPU node for system pods and the Module 9 agent, one RTX 4000
Ada GPU node ($0.52/hr) labeled `pool: gpu` to match the manifest's
nodeSelector, the NVIDIA gpu-operator, and the Linode cloud-firewall
controller.

It does not deploy vLLM. You apply `../manifests/vllm.yaml` with kubectl
afterward. That file stays the single source of truth the modules edit.

The full numbered steps, including teardown, are in
[Path B in the main README](../README.md#path-b-run-it-yourself).

Auth: export `LINODE_TOKEN`. Cost: the GPU node bills hourly whether idle or
busy, so destroy the cluster when you finish.
