"""Prepare five supplementary previews from identified originals; no scientific reduction.

Optional curation utility, not part of CPU data reproduction. Requires the
specified manuscript, mapped research sources, Pillow and Poppler pdftoppm.
No network use and no source writes. Outputs default to build/figure-previews.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from PIL import Image

PDF_SHA256 = "4db81f6350128bd2e6828d93f287ce8067da76010bc3b562451dd22f87366948"
DPI = 300
# PDF points: x0, y0 from upper-left, x1, y1. Includes all axes and legends.
PDF_CROPS = {"FigS2": (44, (120, 136, 490, 362)),
             "FigS7": (48, (120, 47, 490, 270))}
SOURCES = {
    "S3a": ("4.Sweep_Scan_Controls/1.kappa_campaign/Output/Fig_S3_Paul_Tol_muted_2026-10-22/kappa_campaign3.png", "1dcb8c81189f88eab2be76c0663e0161dbd957220720e99164a646a98e45a56e"),
    "S3b": ("4.Sweep_Scan_Controls/1.kappa_campaign/Output/Fig_S3_Paul_Tol_muted_2026-10-22/kappa_campaign4.png", "af37f0f031211e9b2503ce179ad13c770cd0c86c4843f957f0db858332d42136"),
    "FigS5": ("4.Sweep_Scan_Controls/2.kz_campaign/Output/Fig_S5_Paul_Tol_muted_2026-10-21/Fig_S5_Paul_Tol_muted_preview.png", "a9b4e9a046b9d4dcc1f4cc2c4d51369ff0f5293616dc372670dc663a584663b3"),
    "FigS9": ("4.Sweep_Scan_Controls/4.Length_L=30_comparision/Output/Fig_S9_Paul_Tol_muted_2026-10-21/Fig_S9_centered_T40_Paul_Tol_muted_preview.png", "947d6d6a6df5147b5f232072b606ca4e88a7563239c9b679924faf12e91ff13c"),
}


def checked(path: Path, expected: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not a regular source file: {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source SHA256 mismatch: {path.name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/figure-previews"))
    args = parser.parse_args()
    manuscript = checked(args.manuscript, PDF_SHA256)
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Output must be outside the research source root")
    if output == manuscript.parent or manuscript.parent in output.parents:
        raise ValueError("Output must be outside the manuscript source directory")
    source = {}
    for key, (relative, digest) in SOURCES.items():
        path = source_root / relative
        if not path.resolve().is_relative_to(source_root):
            raise ValueError("Source symlink escapes the authorized root")
        source[key] = checked(path, digest)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{name}.png" for name in ["FigS2", "FigS3", "FigS5", "FigS7", "FigS9"]]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing preview")
    poppler = shutil.which("pdftoppm")
    if poppler is None:
        raise RuntimeError("Optional preview extraction requires pdftoppm")
    for name, (page, rect) in PDF_CROPS.items():
        x0, y0 = [math.floor(v * DPI / 72) for v in rect[:2]]
        x1, y1 = [math.ceil(v * DPI / 72) for v in rect[2:]]
        subprocess.run([poppler, "-f", str(page), "-l", str(page),
                        "-singlefile", "-r", str(DPI), "-x", str(x0),
                        "-y", str(y0), "-W", str(x1-x0), "-H", str(y1-y0),
                        "-png", str(manuscript), str(output/name)], check=True)
    with Image.open(source["S3a"]) as first, Image.open(source["S3b"]) as second:
        if first.size != second.size or first.mode != "RGBA" or second.mode != "RGBA":
            raise ValueError("Unexpected original S3 panel layout")
        canvas = Image.new("RGBA", (first.width, first.height + second.height + 40), "white")
        canvas.paste(first, (0, 0))
        canvas.paste(second, (0, first.height + 40))
        # Exact original panel pixels, without resizing or added scientific marks.
        assert canvas.crop((0, 0, first.width, first.height)).tobytes() == first.tobytes()
        assert canvas.crop((0, first.height+40, first.width, first.height+40+second.height)).tobytes() == second.tobytes()
        canvas.save(output/"FigS3.png")
    for name in ["FigS5", "FigS9"]:
        shutil.copyfile(source[name], output/f"{name}.png")
    print("Prepared 5 figure previews; no numerical curves generated or changed.")


if __name__ == "__main__":
    main()
