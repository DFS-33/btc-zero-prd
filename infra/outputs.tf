output "app_service_url" {
  description = "Public HTTPS URL of the deployed API"
  value       = "https://${azurerm_linux_web_app.main.default_hostname}"
}

output "app_service_name" {
  description = "App Service name — use as AZURE_WEBAPP_NAME GitHub Secret"
  value       = azurerm_linux_web_app.main.name
}
