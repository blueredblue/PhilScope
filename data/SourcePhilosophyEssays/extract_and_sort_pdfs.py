#!/usr/bin/env python3
"""
Extract text from all PDFs in a folder, save each as a .txt file,
and print a summary sorted by word/character count.

Usage:
    python3 extract_and_sort_pdfs.py /path/to/your/pdf/folder

If no path is given, it defaults to the current folder.
"""

import sys
import os
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Missing dependency. Install it first with:\n  pip3 install pdfplumber")
    sys.exit(1)


def extract_text(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def main():
    # Force input directory to data/SourcePhilosophyEssays relative to the script
    script_dir = Path(__file__).resolve().parent
    folder = script_dir

    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        sys.exit(0)

    output_dir = folder / "extracted_text"
    output_dir.mkdir(exist_ok=True)
    summary_path = folder / "pdf_word_counts.csv"

    # Load any existing results so we don't lose past runs
    existing_results = {}
    if summary_path.exists():
        import csv
        with open(summary_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_results[row["file"]] = {
                    "file": row["file"],
                    "words": int(row["words"]),
                    "chars": int(row["chars"]),
                }

    results = list(existing_results.values())
    new_count = 0

    for pdf_path in pdf_files:
        txt_path = output_dir / (pdf_path.stem + ".txt")

        # Skip files we've already extracted (incremental mode)
        if pdf_path.name in existing_results and txt_path.exists():
            continue

        print(f"Extracting: {pdf_path.name} ...")
        try:
            text = extract_text(pdf_path)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        # Save extracted text
        txt_path.write_text(text, encoding="utf-8")

        word_count = len(text.split())
        char_count = len(text)

        # Replace any stale entry for this file, then add the fresh one
        results = [r for r in results if r["file"] != pdf_path.name]
        results.append({
            "file": pdf_path.name,
            "words": word_count,
            "chars": char_count,
        })
        new_count += 1

    if not results:
        print("No text could be extracted from any file.")
        return

    if new_count == 0:
        print("No new PDFs found — everything was already extracted.")
    else:
        print(f"\nProcessed {new_count} new file(s).")

    # Sort by word count, descending
    results.sort(key=lambda r: r["words"], reverse=True)

    print("\n" + "=" * 60)
    print(f"{'File':<40}{'Words':>10}{'Chars':>10}")
    print("=" * 60)
    for r in results:
        print(f"{r['file']:<40}{r['words']:>10}{r['chars']:>10}")

    # Also save the summary as a CSV for easy sorting in Excel/Numbers
    import csv
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "words", "chars"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nExtracted text saved to: {output_dir}")
    print(f"Summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()
