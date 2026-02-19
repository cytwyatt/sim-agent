from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def download_open_access_pdf(pdf_url: str, target_path: Path, timeout_seconds: int = 45) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(pdf_url, method="GET")
    req.add_header("User-Agent", "sim-agent/0.1 (+https://local)")
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            data = resp.read()
    except (URLError, HTTPError) as exc:
        raise RuntimeError(f"Failed to download PDF: {exc}") from exc

    if not data.startswith(b"%PDF"):
        raise RuntimeError("Downloaded content does not appear to be a PDF.")

    target_path.write_bytes(data)
    return target_path
