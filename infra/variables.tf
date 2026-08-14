variable "cluster_label" {
  description = "Label for the LKE cluster."
  type        = string
  default     = "own-inference"
}

variable "region" {
  # Must offer RTX 4000 Ada plans. Verify: `linode-cli regions list`
  # or https://www.linode.com/global-infrastructure/availability/
  description = "Linode region with RTX 4000 Ada availability."
  type        = string
  default     = "us-sea"
}

variable "k8s_version" {
  # LKE rotates supported versions over time. Verify the default is still
  # valid: `linode-cli lke versions-list`.
  description = "Kubernetes version supported by LKE."
  type        = string
  default     = "1.35"
}

variable "cpu_node_type" {
  # System pods and the Module 9 agent land here. The vLLM manifest's
  # nodeSelector keeps vLLM off this pool.
  description = "Linode plan for the CPU node pool."
  type        = string
  default     = "g6-standard-4"
}

variable "gpu_node_type" {
  # 1x RTX 4000 Ada (20GB VRAM) tiers (see `linode-cli linodes types | grep gpu`):
  #   g2-gpu-rtx4000a1-s   4 vCPU / 16 GB  / $0.52/hr  (default, fits this
  #                        workshop's models: Qwen3-4B BF16, FP8, 0.6B drafter)
  #   g2-gpu-rtx4000a1-m   8 vCPU / 32 GB  / $0.67/hr
  # The workshop's vLLM Deployment requests 8Gi memory, so the small tier works.
  description = "Linode plan for the GPU node pool."
  type        = string
  default     = "g2-gpu-rtx4000a1-s"
}
