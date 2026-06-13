"""Build the arXiv-ready preprint PDF from docs/paper_{en,ru}.md.

Pipeline: pandoc (md -> standalone HTML + KaTeX) -> headless Chrome print-to-PDF
-> pypdf metadata stamp. This produces a non-TeX PDF, which arXiv accepts as a
direct PDF submission.

Usage:
    uv run --no-sync python scripts/99_build_preprint_pdf.py          # English (default)
    uv run --no-sync python scripts/99_build_preprint_pdf.py --lang ru
    uv run --no-sync python scripts/99_build_preprint_pdf.py --lang both

Output:
    docs/D-MeZO-N_preprint.pdf      (en)
    docs/D-MeZO-N_preprint_ru.pdf   (ru)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Per-language build targets. Cyrillic renders fine in Times New Roman / STIX on Windows.
LANGS = {
    "en": {
        "src": DOCS / "paper_en.md",
        "html": DOCS / "_paper_en_build.html",
        "pdf": DOCS / "D-MeZO-N_preprint.pdf",
        "title": "D-MeZO-N: Decentralized Federated MeZO with Nesterov-Style Stabilization",
        "subject": "Decentralized federated zeroth-order optimization for LLM fine-tuning",
    },
    "ru": {
        "src": DOCS / "paper_ru.md",
        "html": DOCS / "_paper_ru_build.html",
        "pdf": DOCS / "D-MeZO-N_preprint_ru.pdf",
        "title": "D-MeZO-N: Децентрализованный федеративный MeZO с Nesterov-стабилизацией",
        "subject": "Децентрализованная федеративная zeroth-order оптимизация для дообучения LLM",
    },
}

AUTHOR = "Maxim Sukhatsky"

CSS = """
body { font-family: 'Times New Roman', 'STIX Two Text', serif; font-size: 11pt;
       line-height: 1.45; max-width: 17cm; margin: 0 auto; color: #111; }
h1.title { font-size: 17pt; text-align: center; margin-bottom: 4pt; }
p.author, p.date { text-align: center; font-style: italic; font-size: 10pt; margin: 2pt 0; }
h1 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt; }
h2 { font-size: 12pt; margin-top: 14pt; }
h3 { font-size: 11pt; margin-top: 10pt; }
table { border-collapse: collapse; margin: 8pt auto; font-size: 9.5pt; }
th, td { border: 0.5pt solid #888; padding: 3pt 6pt; }
th { background: #f0f0f0; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto; }
figure { margin: 10pt 0; }
figcaption, .caption { font-size: 9pt; color: #333; text-align: justify; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 10pt; color: #333; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f5f5f5; padding: 0 2pt; }
pre { background: #f5f5f5; padding: 6pt; overflow-x: hidden; white-space: pre-wrap; font-size: 9pt; }
.katex { font-size: 1.02em; }
@media print {
  body { max-width: none; margin: 0; }
  h1, h2, h3 { page-break-after: avoid; }
  table, figure, img { page-break-inside: avoid; }
}
"""


def _find_chrome() -> Path:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    chrome = next((c for c in candidates if c.exists()), None)
    if chrome is None:
        sys.exit("[build] Chrome not found — install Chrome or print HTML manually")
    return chrome


def _set_pdf_metadata(pdf: Path, title: str, subject: str) -> None:
    """Stamp author/title into the PDF info dict (arXiv indexers read it)."""
    try:
        import pypdf
    except ImportError:
        print("[build] pypdf not installed — skipping PDF metadata stamp")
        return
    reader = pypdf.PdfReader(str(pdf))
    writer = pypdf.PdfWriter()
    writer.append(reader)
    writer.add_metadata({"/Author": AUTHOR, "/Title": title, "/Subject": subject})
    with pdf.open("wb") as fh:
        writer.write(fh)
    print("[build] PDF metadata stamped (Author/Title/Subject)")


def build(lang: str) -> None:
    cfg = LANGS[lang]
    src, html, pdf = cfg["src"], cfg["html"], cfg["pdf"]
    css_path = DOCS / "_preprint_style.css"
    css_path.write_text(CSS, encoding="utf-8")

    print(f"[build:{lang}] pandoc: {src.name} -> {html.name}")
    subprocess.run(
        [
            "pandoc", str(src),
            "-s", "--katex",
            "--css", css_path.name,
            "--metadata", "document-css=false",
            "-o", str(html),
        ],
        check=True, cwd=DOCS,
    )

    chrome = _find_chrome()
    print(f"[build:{lang}] chrome print-to-pdf -> {pdf.name}")
    subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ],
        check=True,
    )
    _set_pdf_metadata(pdf, cfg["title"], cfg["subject"])
    size_kb = pdf.stat().st_size / 1024
    print(f"[build:{lang}] done: {pdf}  ({size_kb:,.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D-MeZO-N preprint PDF(s).")
    parser.add_argument("--lang", choices=["en", "ru", "both"], default="en")
    args = parser.parse_args()
    langs = ["en", "ru"] if args.lang == "both" else [args.lang]
    for lang in langs:
        build(lang)


if __name__ == "__main__":
    main()
