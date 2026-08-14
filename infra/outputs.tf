output "cluster_id" {
  description = "LKE cluster ID."
  value       = linode_lke_cluster.main.id
}

output "api_endpoints" {
  description = "Cluster API endpoints."
  value       = linode_lke_cluster.main.api_endpoints
}

output "kubeconfig" {
  # base64-encoded. Decode it into a file:
  #   terraform output -raw kubeconfig | base64 -d > kubeconfig.yaml
  description = "Cluster kubeconfig, base64-encoded."
  value       = linode_lke_cluster.main.kubeconfig
  sensitive   = true
}
