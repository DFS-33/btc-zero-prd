variable "app_name" {
  description = "Globally unique Azure App Service name (e.g. passos-magicos-api-abc123)"
  type        = string
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "Brazil South"
}

variable "resource_group_name" {
  description = "Azure Resource Group name"
  type        = string
  default     = "rg-passos-magicos"
}

variable "ghcr_username" {
  description = "GitHub username owning the GHCR package (e.g. DFS-33)"
  type        = string
}

variable "ghcr_pat" {
  description = "GitHub Personal Access Token with read:packages scope"
  type        = string
  sensitive   = true
}
