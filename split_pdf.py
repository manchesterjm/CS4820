"""
PDF Splitter Utility

Splits a large PDF file into two parts for easier reading.

Usage:
    python split_pdf.py

Author: Josh Manchester
Date: November 1, 2025
"""

from pathlib import Path
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        print("ERROR: Please install pypdf or PyPDF2:")
        print("  pip install pypdf")
        print("  or")
        print("  pip install PyPDF2")
        exit(1)


def split_pdf_in_half(input_path: str) -> tuple[str, str]:
    """
    Split a PDF file into two roughly equal parts.

    Args:
        input_path: Path to the input PDF file

    Returns:
        Tuple of (part1_path, part2_path) for the output files
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read the PDF
    print(f"Reading PDF: {input_file.name}")
    reader = PdfReader(str(input_file))
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")

    # Calculate split point (middle of the PDF)
    midpoint = total_pages // 2
    print(f"Splitting at page {midpoint} (Part 1: pages 1-{midpoint}, Part 2: pages {midpoint+1}-{total_pages})")

    # Create output filenames
    base_name = input_file.stem  # filename without extension
    part1_path = input_file.parent / f"{base_name}_Part1.pdf"
    part2_path = input_file.parent / f"{base_name}_Part2.pdf"

    # Create Part 1 (first half)
    print(f"\nCreating Part 1: {part1_path.name}")
    writer1 = PdfWriter()
    for page_num in range(0, midpoint):
        writer1.add_page(reader.pages[page_num])

    with open(part1_path, 'wb') as output_file:
        writer1.write(output_file)
    print(f"  Part 1 saved: {part1_path} ({midpoint} pages)")

    # Create Part 2 (second half)
    print(f"\nCreating Part 2: {part2_path.name}")
    writer2 = PdfWriter()
    for page_num in range(midpoint, total_pages):
        writer2.add_page(reader.pages[page_num])

    with open(part2_path, 'wb') as output_file:
        writer2.write(output_file)
    print(f"  Part 2 saved: {part2_path} ({total_pages - midpoint} pages)")

    print("\nSplit complete!")
    return str(part1_path), str(part2_path)


if __name__ == "__main__":
    # Input PDF file
    input_pdf = r"c:\Users\manch\OneDrive\Desktop\CS4820\8 - Logical Agent Part I-IV.pdf"

    try:
        part1, part2 = split_pdf_in_half(input_pdf)
        print(f"\nOutput files:")
        print(f"  Part 1: {part1}")
        print(f"  Part 2: {part2}")
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
