# -*- coding: utf-8 -*-
"""Generate a handful of starter .cube LUTs into luts/ so the suite works out
of the box. Replace/extend with your own professional .cube files any time --
the suite just reads whatever *.cube it finds in the lut folder.

    python make_sample_luts.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

N = 17  # grid size (17 is a good preview/quality tradeoff)
OUT = Path(__file__).parent / "luts"


def _clip(x):
    return np.clip(x, 0.0, 1.0)


def _grid():
    lin = np.linspace(0.0, 1.0, N, dtype=np.float32)
    r = np.tile(lin, N * N)              # red fastest
    g = np.tile(np.repeat(lin, N), N)
    b = np.repeat(lin, N * N)            # blue slowest
    return r, g, b


def _luma(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---- looks (each returns graded r,g,b in 0..1) ---------------------------- #
def neutral(r, g, b):
    return r, g, b


def warm_film(r, g, b):
    r = _clip(r * 1.06 + 0.012)
    g = _clip(g * 1.01)
    b = _clip(b * 0.92)
    L = _luma(r, g, b)               # gentle S-curve for filmic contrast
    s = _clip(L + (L - 0.5) * 0.18)
    k = np.divide(s, np.maximum(L, 1e-4))
    return _clip(r * k), _clip(g * k), _clip(b * k)


def cool_crisp(r, g, b):
    r = _clip(r * 0.95)
    b = _clip(b * 1.06 + 0.008)
    return _clip((r - 0.5) * 1.12 + 0.5), _clip((g - 0.5) * 1.12 + 0.5), _clip((b - 0.5) * 1.12 + 0.5)


def punchy(r, g, b):
    L = _luma(r, g, b)
    r = L + (r - L) * 1.22           # +saturation
    g = L + (g - L) * 1.22
    b = L + (b - L) * 1.22
    return (_clip((r - 0.5) * 1.18 + 0.5),
            _clip((g - 0.5) * 1.18 + 0.5),
            _clip((b - 0.5) * 1.18 + 0.5))


def teal_orange(r, g, b):
    L = _luma(r, g, b)
    hi, lo = L, 1.0 - L              # highlights warm, shadows teal
    r = r + 0.06 * hi - 0.03 * lo
    g = g + 0.015 * hi + 0.015 * lo
    b = b + 0.07 * lo - 0.03 * hi
    return _clip(r), _clip(g), _clip(b)


def vintage_fade(r, g, b):
    r, g, b = 0.06 + r * 0.9, 0.05 + g * 0.9, 0.045 + b * 0.88   # lifted, faded blacks
    L = _luma(r, g, b)
    r, g, b = L + (r - L) * 0.82, L + (g - L) * 0.82, L + (b - L) * 0.82  # desaturate
    return _clip(r * 1.03 + 0.01), _clip(g), _clip(b * 0.97)


LOOKS = {
    "Neutral": neutral,
    "WarmFilm": warm_film,
    "CoolCrisp": cool_crisp,
    "Punchy": punchy,
    "TealOrange": teal_orange,
    "VintageFade": vintage_fade,
}


def write_cube(name, fn):
    r, g, b = _grid()
    orr, org, orb = fn(r, g, b)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.cube"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"{name}\"\n")
        f.write(f"LUT_3D_SIZE {N}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for i in range(orr.shape[0]):
            f.write(f"{orr[i]:.6f} {org[i]:.6f} {orb[i]:.6f}\n")
    return path


if __name__ == "__main__":
    for nm, fn in LOOKS.items():
        p = write_cube(nm, fn)
        print(f"wrote {p}")
    print(f"\n{len(LOOKS)} starter LUTs in {OUT}")
