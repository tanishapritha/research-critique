from __future__ import annotations

import io
import re
import requests
from typing import List
from pypdf import PdfReader


def fetch_pdf_bytes(url: str, timeout: int = 20) -> bytes:

    if not url.lower().endswith(".pdf"):
        # arXiv entry pages often end with .abs; convert to pdf URL if needed
        # e.g., http://arxiv.org/abs/2001.00001 -> http://arxiv.org/pdf/2001.00001.pdf
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w\.\-]+)", url)
        if m:
            url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def extract_text_from_pdf_bytes(data: bytes, max_pages: int | None = None) -> str:

    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    texts: List[str] = []
    for p in pages:
        try:
            t = p.extract_text() or ""
            texts.append(t)
        except Exception:
            continue
    text = "\n".join(texts)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_text(url: str, max_pages: int | None = None) -> str:

    data = fetch_pdf_bytes(url)
    return extract_text_from_pdf_bytes(data, max_pages=max_pages)
