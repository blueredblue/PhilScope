#!/usr/bin/env python3
"""
Delete PDFs whose word count falls below a threshold, based on
pdf_word_counts.csv (produced by extract_and_sort_pdfs.py).

By default this is a DRY RUN: it only prints what it would delete.
Add --confirm to actually delete the files.

Usage:
    python3 remove_pdfs.py /path/to/your/pdf/folder --min-words 5000
    python3 remove_pdfs.py /path/to/your/pdf/folder --min-words 5000 --confirm
"""

import sys
import csv
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Delete PDFs below a word count threshold.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing the PDFs and pdf_word_counts.csv")
    parser.add_argument("--min-words", type=int, help="Delete files with FEWER words than this")
    parser.add_argument("--max-words", type=int, help="Delete files with MORE words than this")
    parser.add_argument("--zero-words", action="store_true", help="Delete only files with exactly 0 words (usually scanned/image PDFs that failed to extract)")
    parser.add_argument("--confirm", action="store_true", help="Actually delete files (otherwise dry run only)")
    args = parser.parse_args()

    if args.min_words is None and args.max_words is None and not args.zero_words:
        print("Specify --min-words, --max-words, --zero-words, or some combination.")
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    csv_path = folder / "pdf_word_counts.csv"

    if not csv_path.exists():
        print(f"Couldn't find {csv_path}")
        print("Run extract_and_sort_pdfs.py on this folder first.")
        sys.exit(1)

    to_delete = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words = int(row["words"])
            below_min = args.min_words is not None and words < args.min_words
            above_max = args.max_words is not None and words > args.max_words
            is_zero = args.zero_words and words == 0
            if below_min or above_max or is_zero:
                to_delete.append(row)

    if not to_delete:
        print("No matching PDFs found. Nothing to do.")
        return

    range_desc = []
    if args.min_words is not None:
        range_desc.append(f"under {args.min_words} words")
    if args.max_words is not None:
        range_desc.append(f"over {args.max_words} words")
    if args.zero_words:
        range_desc.append("exactly 0 words")
    print(f"Found {len(to_delete)} file(s) {' or '.join(range_desc)}:\n")
    for row in sorted(to_delete, key=lambda r: int(r["words"])):
        print(f"  {row['file']:<40} {row['words']} words")

    if not args.confirm:
        print("\nThis was a DRY RUN — no files were deleted.")
        print("Re-run with --confirm to actually delete them.")
        return

    print("\nDeleting...")
    deleted = 0
    for row in to_delete:
        pdf_path = folder / row["file"]
        txt_path = folder / "extracted_text" / (Path(row["file"]).stem + ".txt")

        if pdf_path.exists():
            pdf_path.unlink()
            deleted += 1
            print(f"  Deleted: {pdf_path.name}")
        if txt_path.exists():
            txt_path.unlink()

    # Rewrite the CSV without the deleted entries
    deleted_names = {row["file"] for row in to_delete}
    with open(csv_path, newline="", encoding="utf-8") as f:
        remaining = [r for r in csv.DictReader(f) if r["file"] not in deleted_names]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "words", "chars"])
        writer.writeheader()
        writer.writerows(remaining)

    print(f"\nDone. Deleted {deleted} PDF(s). CSV updated.")


if __name__ == "__main__":
    main()
