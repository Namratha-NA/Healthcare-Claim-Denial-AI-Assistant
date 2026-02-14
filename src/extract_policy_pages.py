from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "policies"
OUT_FILE = ROOT / "data" / "processed" / "policy_pages.jsonl"


def clean_page_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    written = 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for pdf_path in tqdm(pdfs, desc="Extracting PDF pages"):
            try:
                reader = PdfReader(str(pdf_path))
            except Exception as e:
                print(f"[WARN] Failed to open {pdf_path.name}: {e}")
                continue

            n_pages = len(reader.pages)
            for i in range(n_pages):
                try:
                    raw = reader.pages[i].extract_text() or ""
                except Exception:
                    raw = ""
                text = clean_page_text(raw)

                record = {
                    "doc_id": pdf_path.stem,          # includes __hash
                    "filename": pdf_path.name,
                    "source": "CMS Medicare Claims Processing Manual (PDF)",
                    "page_num": i + 1,               # 1-indexed for humans
                    "extracted_at_utc": now,
                    "text": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"Done. Wrote {written} page records to {OUT_FILE}")


if __name__ == "__main__":
    main()
