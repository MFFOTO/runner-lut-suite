# -*- coding: utf-8 -*-
"""
Convert a graded Hald CLUT image (your look applied to the identity from
make_hald.py) into a .cube LUT.

    python hald_to_cube.py hald_MyLook.png
    python hald_to_cube.py hald_MyLook.png luts/MyLook.cube --size 33
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _trilinear(query: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """Sample cube[r,g,b,3] (S,S,S,3) at query points (M,3) in [0,1]."""
    s = cube.shape[0]
    c = query * (s - 1)
    c0 = np.clip(np.floor(c).astype(np.int64), 0, s - 1)
    c1 = np.clip(c0 + 1, 0, s - 1)
    f = c - c0
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    r0, g0, b0 = c0[:, 0], c0[:, 1], c0[:, 2]
    r1, g1, b1 = c1[:, 0], c1[:, 1], c1[:, 2]

    def T(ri, gi, bi):
        return cube[ri, gi, bi]

    c00 = T(r0, g0, b0) * (1 - fr) + T(r1, g0, b0) * fr
    c10 = T(r0, g1, b0) * (1 - fr) + T(r1, g1, b0) * fr
    c01 = T(r0, g0, b1) * (1 - fr) + T(r1, g0, b1) * fr
    c11 = T(r0, g1, b1) * (1 - fr) + T(r1, g1, b1) * fr
    c0_ = c00 * (1 - fg) + c10 * fg
    c1_ = c01 * (1 - fg) + c11 * fg
    return c0_ * (1 - fb) + c1_ * fb


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a graded Hald image to a .cube LUT")
    ap.add_argument("hald", help="the graded Hald image (your look applied to the identity)")
    ap.add_argument("output", nargs="?", help="output .cube (default: luts/<hald>.cube)")
    ap.add_argument("--size", type=int, default=33, help="output LUT size (default 33)")
    args = ap.parse_args()

    src = Path(args.hald)
    if not src.exists():
        raise SystemExit(f"[ERROR] not found: {src}")
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w != h:
        raise SystemExit(f"[ERROR] {src.name} is {w}x{h}; a Hald image must be square.")
    s = int(round((w * w) ** (1.0 / 3.0)))                 # cube size: s**3 == w*h
    if s ** 3 != w * w:
        raise SystemExit(f"[ERROR] {w}x{h} is not a valid Hald size (need side = level^3).")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    rows = arr.reshape(-1, 3)                               # row-major == red-fastest (Hald order)

    n = int(args.size)
    if n == s:
        out = rows                                         # native size: use as-is (exact)
    else:
        cube = rows.reshape(s, s, s, 3).transpose(2, 1, 0, 3)   # [b,g,r] -> [r,g,b]
        lin = np.linspace(0.0, 1.0, n, dtype=np.float32)
        r = np.tile(lin, n * n)                            # red fastest
        g = np.tile(np.repeat(lin, n), n)
        b = np.repeat(lin, n * n)
        out = _trilinear(np.stack([r, g, b], axis=1), cube)

    out = np.clip(out, 0.0, 1.0)
    dst = Path(args.output) if args.output else (Path("luts") / f"{src.stem}.cube")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{src.stem}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n')
        for row in out:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    print(f"[LUT] {src.name} ({s}^3 Hald) -> {dst}  (LUT_3D_SIZE {n}). Run run_lut.bat and pick it.")


if __name__ == "__main__":
    main()
