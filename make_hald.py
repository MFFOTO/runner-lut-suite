# -*- coding: utf-8 -*-
"""
Generate a neutral Hald CLUT identity image.

Workflow to turn a Lightroom / Camera Raw edit (your XMP looks) into a .cube:
  1. Run this -> hald_identity.png  (a neutral image that encodes every colour).
  2. In Lightroom: import hald_identity.png, copy the Develop settings from an
     edited photo and Paste them onto the Hald (turn OFF any crop/geometry/
     sharpening/noise/vignette -- colour/tone only), then Export it as PNG/TIFF
     (sRGB, no sharpening, no resize).
  3. Run hald_to_cube.py on the exported image -> a .cube of that exact look.

    python make_hald.py            # level 8 -> 512x512 (64^3 LUT), the common size
    python make_hald.py 6 hald6.png
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Hald CLUT identity image")
    ap.add_argument("level", nargs="?", type=int, default=8,
                    help="Hald level: image side = level^3, LUT size = level^2 (default 8)")
    ap.add_argument("output", nargs="?", default="hald_identity.png")
    args = ap.parse_args()

    L = int(args.level)
    S = L * L                      # LUT grid size per axis
    W = L * L * L                  # image side (W*W == S**3)
    idx = np.arange(S * S * S, dtype=np.int64)
    r = idx % S                    # red varies fastest (standard Hald order)
    g = (idx // S) % S
    b = idx // (S * S)
    rgb = np.stack([r, g, b], axis=1).astype(np.float32) / (S - 1)
    img = np.clip(rgb.reshape(W, W, 3) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(args.output)
    print(f"wrote {args.output}  ({W}x{W}, encodes a {S}x{S}x{S} LUT)")
    print("Apply your look to this image in Lightroom (colour/tone only, no crop/"
          "sharpen/resize), export it, then: python hald_to_cube.py <exported>.png")


if __name__ == "__main__":
    main()
