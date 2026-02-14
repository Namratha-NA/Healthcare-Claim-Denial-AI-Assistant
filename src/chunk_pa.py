from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PAGES_FILE = ROOT / "data" / "processed" / "pa_pages.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "chunks"
OUT_FILE = OUT_DIR / "pa_chunks.jsonl"

# Tuneable chunking parameters
TARGET_CHARS = 4200
OVERLAP_CHARS = 300
MIN_CHARS = 450


def normalize(text: str) -> str:
    text = text.strip()
    # Remove repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def iter_page_records() -> Iterator[dict]:
    with PAGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def chunk_text(text: str) -> list[str]:
    """Simple character-based chunking with overlap, respecting paragraph breaks when possible."""
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if len(buf) + len(p) + 2 <= TARGET_CHARS:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            flush_buf()
            # If paragraph itself is huge, hard-split it
            if len(p) > TARGET_CHARS:
                start = 0
                while start < len(p):
                    end = min(start + TARGET_CHARS, len(p))
                    chunks.append(p[start:end].strip())
                    start = max(end - OVERLAP_CHARS, start + 1)
            else:
                buf = p
    flush_buf()

    # Add overlap across chunks (light)
    if OVERLAP_CHARS > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev_tail = chunks[i - 1][-OVERLAP_CHARS:]
                overlapped.append((prev_tail + "\n\n" + c).strip())
        chunks = overlapped
        
    merged = []
    buf = ""
    for c in chunks:
        c = c.strip()
        if not c:
            continue

        # keep adding until we reach ~TARGET_CHARS
        if not buf:
            buf = c
        elif len(buf) + 2 + len(c) <= TARGET_CHARS:
            buf = buf + "\n\n" + c
        else:
            merged.append(buf.strip())
            buf = c

    if buf.strip():
        merged.append(buf.strip())

    chunks = merged

    chunks = [c for c in chunks if len(c.strip()) >= MIN_CHARS]
    return chunks


def main() -> None:
    if not PAGES_FILE.exists():
        raise FileNotFoundError(f"Missing pages file: {PAGES_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_count = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for rec in tqdm(iter_page_records(), desc="Chunking pages"):
            doc_id = rec["doc_id"]
            filename = rec["filename"]
            page_num = rec["page_num"]
            text = normalize(rec.get("text", ""))

            # Skip empty pages
            if len(text) < 50:
                continue

            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks, start=1):
                chunk_id = f"{doc_id}_p{page_num}_c{j}"
                out_rec = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_num": page_num,
                    "source": rec.get("source"),
                    "text": chunk,
                }
                out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(f"Done. Wrote {chunk_count} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
