"""Build the arXiv-ready preprint PDF from docs/paper_en.md.

Pipeline: pandoc (md -> standalone HTML + KaTeX) -> headless Chrome print-to-PDF.
This produces a non-TeX PDF, which arXiv accepts as a direct PDF submission.

Usage:
    uv run --no-sync python scripts/99_build_preprint_pdf.py

Output: docs/D-MeZO-N_preprint.pdf
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SRC = DOCS / "paper_en.md"
HTML = DOCS / "_paper_en_build.html"
PDF = DOCS / "D-MeZO-N_preprint.pdf"

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

def main() -> None:
    css_path = DOCS / "_preprint_style.css"
    css_path.write_text(CSS, encoding="utf-8")

    print(f"[build] pandoc: {SRC.name} -> {HTML.name}")
    subprocess.run(
        [
            "pandoc", str(SRC),
            "-s", "--katex",
            "--css", css_path.name,
            "--metadata", "document-css=false",
            "-o", str(HTML),
        ],
        check=True, cwd=DOCS,
    )

    chrome_candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    chrome = next((c for c in chrome_candidates if c.exists()), None)
    if chrome is None:
        sys.exit("[build] Chrome not found — install Chrome or print HTML manually")

    print(f"[build] chrome print-to-pdf -> {PDF.name}")
    subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ],
        check=True,
    )
    _set_pdf_metadata()
    size_kb = PDF.stat().st_size / 1024
    print(f"[build] done: {PDF}  ({size_kb:,.0f} KB)")


def _set_pdf_metadata() -> None:
    """Stamp author/title into the PDF info dict (arXiv indexers read it)."""
    try:
        import pypdf
    except ImportError:
        print("[build] pypdf not installed — skipping PDF metadata stamp")
        return
    reader = pypdf.PdfReader(str(PDF))
    writer = pypdf.PdfWriter()
    writer.append(reader)
    writer.add_metadata(
        {
            "/Author": "Maxim Sukhatsky",
            "/Title": "D-MeZO-N: Decentralized Federated MeZO with Nesterov-Style Stabilization",
            "/Subject": "Decentralized federated zeroth-order optimization for LLM fine-tuning",
        }
    )
    with PDF.open("wb") as fh:
        writer.write(fh)
    print("[build] PDF metadata stamped (Author/Title/Subject)")


if __name__ == "__main__":
    main()
