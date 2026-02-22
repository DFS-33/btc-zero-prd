"""Trigger Azure App Service to pull the latest container image.

Strategy
--------
Azure App Service exposes a per-app Container Deployment Webhook URL in
Deployment Center.  Posting an empty body to that URL forces an immediate
image pull and container restart — no Service Principal, no SCM Basic Auth,
no OIDC required.  The URL itself contains a time-limited HMAC token issued
by Azure, so it is treated as a secret (AZURE_CONTAINER_WEBHOOK_URL).

Required GitHub secret
----------------------
AZURE_CONTAINER_WEBHOOK_URL
    Obtain from Azure Portal:
      App Service → Deployment Center → Settings tab →
      "Webhook URL" field (bottom of the page).
    Copy the full URL (starts with https://...$<appname>:...) and store it
    as a repository secret.

Optional GitHub secret (still accepted for backward compatibility)
-----------------------------------------------------------------
AZURE_WEBAPP_NAME   — only used in log messages if present.
"""

import os
import sys
import urllib.error
import urllib.request


def post_webhook(url: str) -> int:
    """POST an empty body to *url* and return the HTTP status code."""
    req = urllib.request.Request(
        url,
        data=b"",
        method="POST",
        headers={"Content-Length": "0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            print(f"OK  {status}  {url[:80]}...")
            return status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"ERR {exc.code}  {url[:80]}...  {body[:300]}", file=sys.stderr)
        return exc.code
    except urllib.error.URLError as exc:
        print(f"ERR network  {exc.reason}", file=sys.stderr)
        return -1


def main() -> None:
    webhook_url = os.environ.get("AZURE_CONTAINER_WEBHOOK_URL", "").strip()
    app_name = os.environ.get("AZURE_WEBAPP_NAME", "<unknown>")

    if not webhook_url:
        print(
            "ERROR: AZURE_CONTAINER_WEBHOOK_URL is not set.\n"
            "  1. Open Azure Portal → App Service → Deployment Center.\n"
            "  2. Under 'Settings', copy the 'Webhook URL' field.\n"
            "  3. Store it as the GitHub secret AZURE_CONTAINER_WEBHOOK_URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Triggering container pull for App Service '{app_name}' ...")
    status = post_webhook(webhook_url)

    if status in (200, 202, 204):
        print(f"Container pull triggered successfully (HTTP {status}).")
        sys.exit(0)
    else:
        print(
            f"WARN: Webhook returned HTTP {status}. "
            "The new image is on GHCR but Azure may not have pulled it yet. "
            "You can also restart the app manually from the Azure Portal.",
            file=sys.stderr,
        )
        # Non-fatal: image push to GHCR already succeeded.
        sys.exit(0)


if __name__ == "__main__":
    main()
