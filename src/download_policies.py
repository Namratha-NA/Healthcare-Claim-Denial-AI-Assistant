from __future__ import annotations

import csv
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / "data" / "raw" / "policies" / "policy_seed_urls.txt"
OUT_DIR = ROOT / "data" / "raw" / "policies"
MANIFEST = OUT_DIR / "manifest.csv"


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    base = os.path.basename(parsed.path) or "document.pdf"
    if "." not in base:
        base += ".pdf"
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{Path(base).stem}__{h}{Path(base).suffix}"


def download_file(url: str, out_path: Path, timeout: int = 60) -> tuple[int, int]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ClaimDenialRAG/1.0)"}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        status = r.status_code
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
    return status, bytes_written


def read_urls(seed_file: Path) -> list[str]:
    urls = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    return urls


def main() -> None:
    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = read_urls(SEED_FILE)

    manifest_rows = []
    now = datetime.now(timezone.utc).isoformat()

    for url in tqdm(urls, desc="Downloading CMS policy PDFs"):
        filename = safe_filename_from_url(url)
        out_path = OUT_DIR / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            manifest_rows.append(
                {
                    "downloaded_at_utc": now,
                    "source_url": url,
                    "local_path": str(out_path),
                    "status": "SKIPPED_EXISTS",
                    "http_status": "",
                    "bytes": out_path.stat().st_size,
                }
            )
            continue

        try:
            http_status, nbytes = download_file(url, out_path)
            manifest_rows.append(
                {
                    "downloaded_at_utc": now,
                    "source_url": url,
                    "local_path": str(out_path),
                    "status": "DOWNLOADED",
                    "http_status": http_status,
                    "bytes": nbytes,
                }
            )
        except Exception as e:
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass
            manifest_rows.append(
                {
                    "downloaded_at_utc": now,
                    "source_url": url,
                    "local_path": str(out_path),
                    "status": f"FAILED: {type(e).__name__}",
                    "http_status": "",
                    "bytes": 0,
                }
            )

    write_header = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "downloaded_at_utc",
                "source_url",
                "local_path",
                "status",
                "http_status",
                "bytes",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Files saved to: {OUT_DIR}")
    print(f"Manifest updated: {MANIFEST}")


if __name__ == "__main__":
    main()
