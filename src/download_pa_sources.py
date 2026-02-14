from __future__ import annotations

import hashlib
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "raw" / "prior_auth" / "pa_seed_urls.txt"
OUT_DIR = ROOT / "data" / "raw" / "prior_auth" / "docs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def sha(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

def save_pdf(url: str, content: bytes):
    name = Path(url.split("?")[0]).name
    out = OUT_DIR / f"{name[:-4]}__{sha(url)}.pdf"
    out.write_bytes(content)

def save_html(url: str, content: bytes):
    out = OUT_DIR / f"cms_pa__{sha(url)}.html"
    out.write_bytes(content)

def main():
    urls = [u.strip() for u in SEED.read_text(encoding="utf-8").splitlines() if u.strip()]
    for url in tqdm(urls, desc="Downloading PA sources"):
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()

        if url.lower().endswith(".pdf") or "application/pdf" in ctype:
            save_pdf(url, r.content)
        else:
            save_html(url, r.content)

    print(f"Saved files to: {OUT_DIR}")

if __name__ == "__main__":
    main()
