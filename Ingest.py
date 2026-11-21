import json
from pathlib import Path
from pypdf import PdfReader

DOCS_DIR = Path("Data")
OUTPUT_FILE = Path("chunks.json")
CHUNK_SIZE = 800
OVERLAP = 150

def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces

chunks = []
for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i, piece in enumerate(split_text(text)):
            chunks.append({
                "id": f"{pdf_path.stem}-p{page_number}-{i}",
                "document": pdf_path.name,
                "page": page_number,
                "text": piece,
            })

OUTPUT_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")