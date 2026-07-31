# -*- coding: utf-8 -*-
"""
Compare .cube LUTs: report whether any two are identical and characterise the
look each one imparts (white balance, black lift, white point, contrast,
saturation). Handy to confirm two exported LUTs really are different grades.

    python compare_luts.py luts/Look1.cube luts/Look2.cube
    python compare_luts.py                 # compares every .cube in luts/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import lut_suite as L

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _apply(lut, rgb01: np.ndarray) -> np.ndarray:
    u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8).reshape(1, -1, 3)
    return L.apply_lut(u8, lut, 1.0).reshape(-1, 3).astype(np.float32) / 255.0


def characterize(lut) -> dict:
    gray = np.stack([np.linspace(0, 1, 33, dtype=np.float32)] * 3, axis=1)
    og = _apply(lut, gray)
    mid = og[len(og) // 2]
    swatch = np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7],
                       [0.7, 0.7, 0.15], [0.15, 0.7, 0.7], [0.7, 0.15, 0.7]], dtype=np.float32)
    os = _apply(lut, swatch)

    def sat(x):
        mx, mn = x.max(axis=1), x.min(axis=1)
        return float(np.mean((mx - mn) / np.clip(mx, 1e-6, None)))

    return {
        "black_lift": float(og[0].mean()),
        "white": float(og[-1].mean()),
        "contrast": float(og[-1].mean() - og[0].mean()),
        "wb_R-B": float(mid[0] - mid[2]),
        "wb_R-G": float(mid[0] - mid[1]),
        "sat_x": (sat(os) / max(sat(swatch), 1e-6)),
    }


def grid(lut, n: int = 17) -> np.ndarray:
    lin = np.linspace(0, 1, n, dtype=np.float32)
    r = np.tile(lin, n * n); g = np.tile(np.repeat(lin, n), n); b = np.repeat(lin, n * n)
    return _apply(lut, np.stack([r, g, b], axis=1))


def main() -> None:
    args = sys.argv[1:]
    pairwise = "--pairwise" in args
    args = [a for a in args if a != "--pairwise"]
    paths = [Path(a) for a in args] if args else sorted(Path("luts").glob("*.cube"))
    paths = [p for p in paths if p.exists()]
    if len(paths) < 1:
        print("No .cube files given or found in luts/."); return

    luts, names, grids = [], [], []
    print("=" * 68)
    for p in paths:
        try:
            lut = L.parse_cube(p)
        except Exception as exc:
            print(f"[skip] {p.name}: {exc}"); continue
        c = characterize(lut)
        luts.append(lut); names.append(p.name); grids.append(grid(lut))
        cast = ("warm/red" if c["wb_R-B"] > 0.01 else "cool/blue" if c["wb_R-B"] < -0.01 else "neutral")
        print(f"{p.name}")
        print(f"   black-lift {c['black_lift']:.2f}  white {c['white']:.2f}  contrast {c['contrast']:.2f}  "
              f"saturation x{c['sat_x']:.2f}  WB {cast} (R-B {c['wb_R-B']:+.02f})")
    print("=" * 68)

    if len(grids) < 2:
        return

    # Cluster LUTs that are the same grade (mean abs diff < 0.5/255).
    thr = 0.5
    groups: list[list[int]] = []
    for i in range(len(grids)):
        for grp in groups:
            if float(np.abs(grids[i] - grids[grp[0]]).mean()) * 255 < thr:
                grp.append(i)
                break
        else:
            groups.append([i])

    dups = [g for g in groups if len(g) > 1]
    print(f"{len(grids)} LUT(s) compared -> {len(groups)} unique look(s).")
    if dups:
        print("\nIDENTICAL groups (same grade -- keep one of each, the rest are duplicates):")
        for g in dups:
            print("   " + "  ==  ".join(names[k] for k in g))
    else:
        print("No duplicates -- every LUT is a distinct grade.")

    if pairwise:
        print("\nFull pairwise (mean abs /255):")
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                d = float(np.abs(grids[i] - grids[j]).mean()) * 255
                print(f"   {names[i]}  vs  {names[j]}:  {d:.1f}" + ("  IDENTICAL" if d < thr else ""))


if __name__ == "__main__":
    main()
