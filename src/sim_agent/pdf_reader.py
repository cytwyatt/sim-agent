from __future__ import annotations

from pathlib import Path


def extract_pdf_text(pdf_path: Path, max_pages: int = 30, max_chars: int = 80000) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    text_chunks: list[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            text_chunks.append(page_text)
        if sum(len(c) for c in text_chunks) > max_chars:
            break

    text = "\n".join(text_chunks)
    return text[:max_chars]
