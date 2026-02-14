from __future__ import annotations

import json, re
from datetime import datetime, timezone
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "raw" / "prior_auth" / "docs"
OUT_FILE = ROOT / "data" / "processed" / "pa_pages.jsonl"

def clean(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pdf(path: Path):
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        yield i, clean(page.extract_text() or "")

def extract_html(path: Path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # remove nav/script/style
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = clean(soup.get_text("\n"))
    # treat as one "page"
    yield 1, text

def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    files = sorted([p for p in DOC_DIR.iterdir() if p.is_file()])
    written = 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for p in tqdm(files, desc="Extracting PA docs"):
            if p.suffix.lower() == ".pdf":
                iterator = extract_pdf(p)
                source = "CMS Prior Authorization (PDF)"
            else:
                iterator = extract_html(p)
                source = "CMS Prior Authorization (HTML)"

            for page_num, text in iterator:
                if not text:
                    continue
                rec = {
                    "doc_id": p.stem,
                    "filename": p.name,
                    "source": source,
                    "page_num": page_num,
                    "extracted_at_utc": now,
                    "text": text,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. Wrote {written} records to {OUT_FILE}")

if __name__ == "__main__":
    main()
