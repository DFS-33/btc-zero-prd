"""Trigger Azure App Service to pull the latest container image via Kudu API."""
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
import base64
import os
import sys


def kudu_post(url, creds):
    auth = base64.b64encode(creds.encode()).decode()
    req = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            print(f"OK  {r.status}  {url}")
            return r.status
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"ERR {e.code}  {url}  {body[:300]}", file=sys.stderr)
        return e.code


profile_xml = os.environ["AZURE_PUBLISH_PROFILE"]
app_name = os.environ["AZURE_WEBAPP_NAME"]

root = ET.fromstring(profile_xml)
prof = root.find('.//publishProfile[@publishMethod="MSDeploy"]')
if prof is None:
    print("ERROR: MSDeploy profile not found in publish profile", file=sys.stderr)
    sys.exit(1)

user = prof.get("userName")
pwd = prof.get("userPWD")
creds = f"{user}:{pwd}"

base = f"https://{app_name}.scm.azurewebsites.net"

# Try container webhook first (triggers fresh image pull), fall back to restart.
# Non-fatal: image is already pushed to GHCR; Azure will pull on next restart.
code = kudu_post(f"{base}/api/registry/webhook", creds)
if code not in (200, 204):
    code = kudu_post(f"{base}/api/restart", creds)
    if code not in (200, 204):
        print(
            "WARN: Kudu trigger failed — image is on GHCR. "
            "Run 'terraform apply' or restart from Azure Portal.",
            file=sys.stderr,
        )
