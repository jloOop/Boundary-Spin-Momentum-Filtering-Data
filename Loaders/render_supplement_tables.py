#!/usr/bin/env python3
"""Render the supplementary table catalog from canonical CSVs, or check freshness."""
import argparse
from pathlib import Path
from centered.supplement_tables import CATALOG, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed catalog is stale; write nothing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / CATALOG
    text = render(root)
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            raise SystemExit("Stale supplementary catalog: run python Loaders/render_supplement_tables.py")
        print("Supplementary table catalog matches canonical CSVs and declared display rounding")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"Rendered {target.relative_to(root)}; numerical inputs unchanged")


if __name__ == "__main__":
    main()
