# -*- coding: utf-8 -*-
"""
Generate a .cube LUT from a BEFORE/AFTER image pair.

Given the same photo un-graded (ORIGINAL) and graded (GRADED), this fits a
polynomial colour transform original_rgb -> graded_rgb and bakes it into a 3D
LUT. Unlike make_lut_from_image.py (which estimates a look from statistics),
this reproduces the *actual* edit -- exposure, white balance, contrast/tone
curve, saturation, HSL and split-tone all captured together, because they're
all baked into the after image's pixels.

Requirement: the two images must be the SAME framing (same crop/geometry, only
colour differs). Any resolution is fine -- they're resized to a common size.

    python make_lut_from_pair.py original.jpg graded.jpg
    python make_lut_from_pair.py original.jpg graded.jpg luts/MyGrade.cube --degree 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def poly_features(rgb: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial feature matrix for (M,3) RGB in [0,1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    o = np.ones_like(r)
    feats = [o, r, g, b]
    if degree >= 2:
        feats += [r * r, g * g, b * b, r * g, r * b, g * b]
    if degree >= 3:
        feats += [r * r * r, g * g * g, b * b * b,
                  r * r * g, r * r * b, g * g * r, g * g * b, b * b * r, b * b * g,
                  r * g * b]
    return np.stack(feats, axis=1).astype(np.float32)


def fit_transform(orig: np.ndarray, graded: np.ndarray, degree: int, ridge: float = 1e-3) -> np.ndarray:
    """Ridge least-squares W so that poly_features(orig) @ W ~= graded."""
    X = poly_features(orig, degree)
    A = X.T @ X + ridge * np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(A, X.T @ graded).astype(np.float32)   # (F, 3)


def _load(path: Path, max_edge: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a .cube LUT from a before/after image pair")
    ap.add_argument("original", help="the un-graded image (before)")
    ap.add_argument("graded", help="the graded image (after) -- same framing")
    ap.add_argument("output", nargs="?", help="output .cube (default: luts/<graded>.cube)")
    ap.add_argument("--degree", type=int, default=2, choices=(1, 2, 3), help="fit degree (default 2)")
    ap.add_argument("--size", type=int, default=33, help="LUT grid size (default 33)")
    ap.add_argument("--sample", type=int, default=640, help="analysis long-edge px (default 640)")
    args = ap.parse_args()

    op, gp = Path(args.original), Path(args.graded)
    for p in (op, gp):
        if not p.exists():
            raise SystemExit(f"[ERROR] not found: {p}")

    o_img = _load(op, args.sample)
    g_img = _load(gp, args.sample)
    if o_img.size != g_img.size:                       # align to a common grid
        g_img = g_img.resize(o_img.size, Image.LANCZOS)
    orig = (np.asarray(o_img, dtype=np.float32) / 255.0).reshape(-1, 3)
    graded = (np.asarray(g_img, dtype=np.float32) / 255.0).reshape(-1, 3)

    W = fit_transform(orig, graded, args.degree)
    pred = np.clip(poly_features(orig, args.degree) @ W, 0.0, 1.0)
    rmse = float(np.sqrt(np.mean((pred - graded) ** 2)))
    print(f"[LUT] fit degree {args.degree}: RMSE {rmse * 255:.1f}/255 over {orig.shape[0]} px "
          f"({'excellent' if rmse*255 < 4 else 'good' if rmse*255 < 9 else 'rough -- try --degree 3 or check alignment'})")

    n = int(args.size)
    lin = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r = np.tile(lin, n * n)                             # red fastest (cube order)
    g = np.tile(np.repeat(lin, n), n)
    b = np.repeat(lin, n * n)
    grid = np.stack([r, g, b], axis=1)
    out = np.clip(poly_features(grid, args.degree) @ W, 0.0, 1.0)

    dst = Path(args.output) if args.output else (Path("luts") / f"{gp.stem}.cube")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{gp.stem}"\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n')
        for row in out:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    print(f"[LUT] wrote {dst}  (LUT_3D_SIZE {n}). Run run_lut.bat and pick it.")


if __name__ == "__main__":
    main()
