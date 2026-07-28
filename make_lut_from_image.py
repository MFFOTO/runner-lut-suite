# -*- coding: utf-8 -*-
"""
Generate a .cube LUT that imparts the *look* of a reference image.

It analyses the reference's global colour character -- white balance / colour
cast, tonal range (lifted or crushed blacks, muted or bright whites), midtone
gamma, saturation, and shadow/highlight split-tone -- and bakes that into a 3D
LUT. Drop the result into luts/ and the LUT suite applies it to a whole folder.

    python make_lut_from_image.py hero.jpg
    python make_lut_from_image.py hero.jpg luts/MyLook.cube --strength 0.8

Note: this extracts a *grade* (a consistent look), not a pixel-perfect copy of
the image. Point it at a shot whose colour treatment you like -- a flat/hazy
snapshot yields a flat/hazy LUT.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def analyse(ref: np.ndarray) -> dict:
    """ref: HxWx3 float32 in [0,1]. Returns look parameters."""
    flat = ref.reshape(-1, 3)
    lum = flat @ _LUMA

    means = flat.mean(axis=0)                                  # colour balance
    gray = float(means.mean())
    wb = means / max(gray, 1e-6)                               # gains to impart the cast

    lo = float(np.percentile(lum, 2))                          # black point
    hi = float(np.percentile(lum, 98))                         # white point
    mid = float(np.percentile(lum, 50))
    midn = float(np.clip((mid - lo) / max(hi - lo, 1e-6), 0.02, 0.98))
    gamma = float(np.log(midn) / np.log(0.5))                  # input**gamma maps 0.5 -> midn

    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    sat_mean = float(((mx - mn) / np.clip(mx, 1e-6, None)).mean())   # mean HSV saturation

    def tint(mask: np.ndarray) -> np.ndarray:
        if int(mask.sum()) < 50:
            return np.zeros(3, dtype=np.float32)
        px = flat[mask]
        return (px - px.mean(axis=1, keepdims=True)).mean(axis=0)     # avg chroma vector

    return {
        "wb": wb.astype(np.float32),
        "lo": lo, "hi": hi, "gamma": gamma, "sat_mean": sat_mean,
        "shadow": tint(lum < 0.35), "highlight": tint(lum > 0.65),
    }


def build_lut(a: dict, size: int, strength: float, sat_base: float = 0.30) -> np.ndarray:
    """Return the graded grid rows (N^3, 3), red varying fastest (cube order)."""
    lin = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r = np.tile(lin, size * size)                    # red fastest
    g = np.tile(np.repeat(lin, size), size)
    b = np.repeat(lin, size * size)                  # blue slowest
    ident = np.stack([r, g, b], axis=1)
    out = ident.copy()

    # 1) white balance -- normalised so it tints without shifting overall brightness
    out *= (a["wb"] / max(float(a["wb"].mean()), 1e-6))
    out = np.clip(out, 0.0, 1.0)
    # 2) midtone gamma, then map into the reference's tonal range (lift/crush)
    out = out ** a["gamma"]
    out = a["lo"] + out * (a["hi"] - a["lo"])
    # 3) saturation toward the reference
    lum = (out @ _LUMA)[:, None]
    out = lum + (out - lum) * float(np.clip(a["sat_mean"] / sat_base, 0.4, 1.6))
    # 4) split-tone: bias shadows / highlights by their measured chroma
    ln = np.clip(out @ _LUMA, 0.0, 1.0)[:, None]
    out = out + a["shadow"] * (1.0 - ln) * 0.5 + a["highlight"] * ln * 0.5

    out = np.clip(out, 0.0, 1.0)
    out = ident * (1.0 - strength) + out * strength          # overall strength dial
    return np.clip(out, 0.0, 1.0)


def write_cube(path: Path, rows: np.ndarray, size: int, title: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for row in rows:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a .cube LUT from a reference image's look")
    ap.add_argument("image", help="reference image (jpg/png/...) whose look you want")
    ap.add_argument("output", nargs="?", help="output .cube (default: luts/<image>.cube)")
    ap.add_argument("--strength", type=float, default=1.0, help="0..1 look intensity (default 1.0)")
    ap.add_argument("--size", type=int, default=33, help="LUT grid size (default 33)")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.exists():
        raise SystemExit(f"[ERROR] image not found: {src}")
    img = Image.open(src).convert("RGB")
    img.thumbnail((512, 512), Image.LANCZOS)                  # analyse a downscaled copy (fast)
    ref = np.asarray(img, dtype=np.float32) / 255.0

    a = analyse(ref)
    print(f"[LUT] {src.name}: wb(R,G,B)={a['wb'].round(2).tolist()}  "
          f"blacks={a['lo']:.2f} whites={a['hi']:.2f} gamma={a['gamma']:.2f}  "
          f"sat={a['sat_mean']:.2f}")
    rows = build_lut(a, size=int(args.size), strength=float(args.strength))

    out = Path(args.output) if args.output else (Path("luts") / f"{src.stem}.cube")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_cube(out, rows, int(args.size), title=src.stem)
    print(f"[LUT] wrote {out}  (LUT_3D_SIZE {args.size}, strength {args.strength})")
    print("      Now run run_lut.bat and pick it, or preview it against your crops.")


if __name__ == "__main__":
    main()
