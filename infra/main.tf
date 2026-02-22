terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
  # Auth via Azure CLI: run `az login` before terraform apply
  # No client_id / client_secret needed
}

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_service_plan" "main" {
  name                = "asp-passos-magicos"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "F1"
}

resource "azurerm_linux_web_app" "main" {
  name                = var.app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_service_plan.main.location
  service_plan_id     = azurerm_service_plan.main.id

  site_config {
    always_on = false # Required: F1 does not support always_on = true

    application_stack {
      docker_image_name   = "dfs-33/btc-zero-prd:latest"
      docker_registry_url = "https://ghcr.io"
      # No credentials needed — GHCR package is public
    }
  }

  app_settings = {
    WEBSITES_PORT = "8000"
  }
}
