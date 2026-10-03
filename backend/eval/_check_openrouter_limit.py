"""Report OpenRouter key limit/usage without printing the key."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def _load_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    for path in (Path("backend/.env"), Path(".env")):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> int:
    key = _load_key()
    if not key:
        print("NO_KEY")
        return 1
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode())
    data = payload.get("data") or payload
    print(
        json.dumps(
            {
                "label": data.get("label"),
                "limit": data.get("limit"),
                "limit_remaining": data.get("limit_remaining"),
                "usage": data.get("usage"),
                "usage_daily": data.get("usage_daily"),
                "usage_weekly": data.get("usage_weekly"),
                "usage_monthly": data.get("usage_monthly"),
                "is_free_tier": data.get("is_free_tier"),
                "rate_limit": data.get("rate_limit"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
