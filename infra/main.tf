# Path B paved path: the cluster this workshop's manifests expect, from zero.
#
# What this provisions:
#   - An LKE cluster with a small CPU node pool for system workloads and the
#     Module 9 agent.
#   - A GPU node pool (one RTX 4000 Ada), labeled `pool=gpu` so the
#     nodeSelector in ../manifests/vllm.yaml matches.
#   - The NVIDIA gpu-operator (drivers + device plugin). Nodes report
#     `nvidia.com/gpu` 3-5 minutes after the cluster is up.
#   - The Linode cloud-firewall-controller, which firewalls each worker node
#     with the LKE-recommended ruleset so the public NodePort range is not
#     reachable from the internet.
#
# What this does NOT provision:
#   - The vLLM workload. Apply ../manifests/vllm.yaml with kubectl after the
#     cluster is up. That file stays the single source of truth the modules
#     edit, so terraform never fights the workshop over it.
#
# Auth: export LINODE_TOKEN before running.
# Cost: the GPU node bills hourly whether idle or busy. `terraform destroy`
# when you finish, after deleting the workload (see the README teardown step).

provider "linode" {
  # Auth: set LINODE_TOKEN in your environment.
}

# Pull the cluster's kubeconfig for use by the helm provider below.
locals {
  kubeconfig = yamldecode(base64decode(linode_lke_cluster.main.kubeconfig))
}

# Helm provider talks to the LKE cluster's API server using the kubeconfig
# extracted from the LKE resource above. Only used to install bootstrap
# operators, not workloads.
provider "helm" {
  kubernetes {
    host                   = local.kubeconfig.clusters[0].cluster.server
    cluster_ca_certificate = base64decode(local.kubeconfig.clusters[0].cluster["certificate-authority-data"])
    token                  = local.kubeconfig.users[0].user.token
  }
}

resource "linode_lke_cluster" "main" {
  label       = var.cluster_label
  region      = var.region
  k8s_version = var.k8s_version

  # System pool. vLLM does NOT land here (no GPU). kube-system, the
  # gpu-operator control pods, and the Module 9 agent run on this node.
  pool {
    type  = var.cpu_node_type
    count = 1
  }
}

# GPU pool, defined separately so it can be scaled or replaced without
# touching the cluster resource.
resource "linode_lke_node_pool" "gpu" {
  cluster_id = linode_lke_cluster.main.id
  type       = var.gpu_node_type
  node_count = 1

  # Kubernetes node label propagated from the Linode side. The vLLM
  # Deployment uses `nodeSelector: pool=gpu` to schedule onto this node.
  labels = {
    pool = "gpu"
  }
}

# NVIDIA gpu-operator: installs the driver and device plugin so the GPU node
# advertises `nvidia.com/gpu`. Until it finishes, the vLLM pod stays Pending.
resource "helm_release" "gpu_operator" {
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  namespace        = "gpu-operator"
  create_namespace = true
  wait             = true
  timeout          = 900

  depends_on = [linode_lke_cluster.main, linode_lke_node_pool.gpu]
}

# Cloud Firewall Controller: puts a Linode Cloud Firewall on every worker node
# with the LKE-recommended ruleset (allows control-plane and intra-cluster
# traffic, drops everything else, including the public NodePort range).
# CRDs first, then the controller.
resource "helm_release" "cloud_firewall_crd" {
  name             = "cloud-firewall-crd"
  repository       = "https://linode.github.io/cloud-firewall-controller"
  chart            = "cloud-firewall-crd"
  namespace        = "kube-system"
  create_namespace = false
  wait             = true
  timeout          = 300

  depends_on = [linode_lke_cluster.main, linode_lke_node_pool.gpu]
}

resource "helm_release" "cloud_firewall_controller" {
  name             = "cloud-firewall"
  repository       = "https://linode.github.io/cloud-firewall-controller"
  chart            = "cloud-firewall-controller"
  namespace        = "kube-system"
  create_namespace = false
  wait             = true
  timeout          = 300

  depends_on = [helm_release.cloud_firewall_crd]
}
