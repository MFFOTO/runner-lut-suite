# -*- coding: utf-8 -*-
"""
Runner LUT Suite  --  local color-grading pass for a folder of images.

Two stages:
  1. PREVIEW / SELECT  -- applies every .cube LUT in `luts/` to a few sample
     images, writes side-by-side preview sheets you can open, and lets you pick
     one interactively. Loop until you're happy.
  2. BATCH             -- applies the chosen LUT to the entire input folder,
     in parallel across CPU cores (fast: LUT + JPEG only, no GPU needed).

Runs entirely on CPU on the local machine. Point it at a folder of finished
crops; it does not upscale or touch faces -- just the final look.

Usage:
    python lut_suite.py                     # interactive, uses settings_lut.json
    python lut_suite.py --input D:/crops --output D:/graded
    python lut_suite.py --lut warmfilm.cube --yes    # skip prompts (batch now)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "input_folder": "input",
        "output_folder": "output_lut",
        "lut_folder": "luts",
        "preview_folder": None,   # None -> <output_folder>/_previews
    },
    "lut": {
        "selected": None,     # filename in lut_folder; set to skip selection
        "strength": 1.0,      # 0..1 blend of graded over original
    },
    "output": {
        "jpeg_quality": 95,
        "workers": 0,         # 0 = all CPU cores
        "suffix": "",         # appended to output filename stem
        "preview_max_edge": 640,
        "sample_count": 4,
    },
    "options": {"recursive": True, "auto_open_previews": True},
}


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {path} not found -- using built-in defaults.")
        return json.loads(json.dumps(DEFAULTS))
    with open(p, "r", encoding="utf-8") as f:
        return _deep_merge(DEFAULTS, json.load(f))


# --------------------------------------------------------------------------- #
#  .cube parsing
# --------------------------------------------------------------------------- #
def parse_cube(path: Path) -> Dict[str, Any]:
    """Parse an Adobe .cube LUT (1D or 3D). Returns a dict with the lookup
    table already oriented as table[i_r, i_g, i_b] (3D) or curve[n,3] (1D)."""
    size: Optional[int] = None
    dim = 3
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    title = None
    data: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            u = s.upper()
            if u.startswith("TITLE"):
                bits = s.split(None, 1)
                title = bits[1].strip().strip('"') if len(bits) > 1 else None
                continue
            if u.startswith("LUT_3D_SIZE"):
                size, dim = int(s.split()[1]), 3
                continue
            if u.startswith("LUT_1D_SIZE"):
                size, dim = int(s.split()[1]), 1
                continue
            if u.startswith("DOMAIN_MIN"):
                dmin = [float(x) for x in s.split()[1:4]]
                continue
            if u.startswith("DOMAIN_MAX"):
                dmax = [float(x) for x in s.split()[1:4]]
                continue
            if u.startswith("LUT_3D_INPUT_RANGE") or u.startswith("LUT_1D_INPUT_RANGE"):
                v = [float(x) for x in s.split()[1:3]]
                dmin, dmax = [v[0]] * 3, [v[1]] * 3
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    if size is None:
        raise ValueError(f"{path.name}: missing LUT_3D_SIZE / LUT_1D_SIZE")
    arr = np.asarray(data, dtype=np.float32)
    out = {"dim": dim, "size": size, "title": title,
           "dmin": np.asarray(dmin, dtype=np.float32),
           "dmax": np.asarray(dmax, dtype=np.float32)}
    if dim == 3:
        if arr.shape[0] != size ** 3:
            raise ValueError(f"{path.name}: expected {size**3} rows, got {arr.shape[0]}")
        # rows iterate red fastest -> reshape gives [b,g,r], transpose to [r,g,b]
        out["table"] = arr.reshape(size, size, size, 3).transpose(2, 1, 0, 3).copy()
    else:
        if arr.shape[0] != size:
            raise ValueError(f"{path.name}: expected {size} rows, got {arr.shape[0]}")
        out["curve"] = arr
    return out


# --------------------------------------------------------------------------- #
#  LUT application
# --------------------------------------------------------------------------- #
def _trilinear(rgb01: np.ndarray, table: np.ndarray) -> np.ndarray:
    """rgb01: (M,3) in [0,1]; table: (N,N,N,3) indexed [i_r,i_g,i_b]."""
    n = table.shape[0]
    scale = n - 1
    c = rgb01 * scale
    c0 = np.clip(np.floor(c).astype(np.int64), 0, n - 1)
    c1 = np.clip(c0 + 1, 0, n - 1)
    f = c - c0
    fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    r0, g0, b0 = c0[:, 0], c0[:, 1], c0[:, 2]
    r1, g1, b1 = c1[:, 0], c1[:, 1], c1[:, 2]

    def T(ri, gi, bi):
        return table[ri, gi, bi]

    c00 = T(r0, g0, b0) * (1 - fr) + T(r1, g0, b0) * fr
    c10 = T(r0, g1, b0) * (1 - fr) + T(r1, g1, b0) * fr
    c01 = T(r0, g0, b1) * (1 - fr) + T(r1, g0, b1) * fr
    c11 = T(r0, g1, b1) * (1 - fr) + T(r1, g1, b1) * fr
    c0_ = c00 * (1 - fg) + c10 * fg
    c1_ = c01 * (1 - fg) + c11 * fg
    return c0_ * (1 - fb) + c1_ * fb


def _curve1d(rgb01: np.ndarray, curve: np.ndarray) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, curve.shape[0], dtype=np.float32)
    out = np.empty_like(rgb01)
    for ch in range(3):
        out[:, ch] = np.interp(rgb01[:, ch], xs, curve[:, ch])
    return out


def apply_lut(img_uint8: np.ndarray, lut: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Apply a parsed LUT to an HxWx3 uint8 RGB image. Chunked to bound memory."""
    rgb = img_uint8.astype(np.float32) / 255.0
    span = np.maximum(lut["dmax"] - lut["dmin"], 1e-8)
    norm = np.clip((rgb - lut["dmin"]) / span, 0.0, 1.0)
    flat = norm.reshape(-1, 3)
    out = np.empty_like(flat)
    chunk = 1_000_000
    for i in range(0, flat.shape[0], chunk):
        blk = flat[i:i + chunk]
        out[i:i + chunk] = _trilinear(blk, lut["table"]) if lut["dim"] == 3 else _curve1d(blk, lut["curve"])
    graded = np.clip(out.reshape(rgb.shape), 0.0, 1.0)
    if strength < 1.0:
        graded = rgb * (1.0 - strength) + graded * strength
    return (graded * 255.0 + 0.5).astype(np.uint8)


def build_dense_lut(lut: Dict[str, Any], strength: float = 1.0) -> np.ndarray:
    """Pre-expand a LUT into a dense 256x256x256x3 uint8 table (one-time cost).
    Batch then applies it as a single array gather -- ~3.5x faster than
    interpolating every image, since image pixels are uint8 anyway."""
    r = np.arange(256, dtype=np.uint8)
    R, G, B = np.meshgrid(r, r, r, indexing="ij")
    grid = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    return apply_lut(grid, lut, strength).reshape(256, 256, 256, 3)


def apply_dense(img_uint8: np.ndarray, dense: np.ndarray) -> np.ndarray:
    """Apply a pre-built dense LUT: dense[r, g, b] for every pixel."""
    return dense[img_uint8[..., 0], img_uint8[..., 1], img_uint8[..., 2]]


# --------------------------------------------------------------------------- #
#  Preview sheets
# --------------------------------------------------------------------------- #
def _thumb(img, max_edge):
    from PIL import Image
    w, h = img.size
    s = max_edge / float(max(w, h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS) if s < 1 else img.copy()


def _label(img, text):
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 4
    tw = draw.textlength(text, font=font) if hasattr(draw, "textlength") else 8 * len(text)
    draw.rectangle([0, 0, tw + 2 * pad, 16], fill=(0, 0, 0))
    draw.text((pad, 2), text, fill=(255, 255, 255), font=font)
    return img


def _montage(tiles, cols, bg=(24, 24, 24)):
    from PIL import Image
    if not tiles:
        return Image.new("RGB", (16, 16), bg)
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    gap = 6
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (gap + c * (tw + gap), gap + r * (th + gap)))
    return sheet


def build_overview(sample, luts, names, max_edge, out_path):
    from PIL import Image
    base = _thumb(sample, max_edge // 2)
    tiles = [_label(base.copy(), "ORIGINAL")]
    for lut, name in zip(luts, names):
        graded = Image.fromarray(apply_lut(np.asarray(base.convert("RGB")), lut, 1.0))
        tiles.append(_label(graded, name))
    cols = int(np.ceil(np.sqrt(len(tiles))))
    _montage(tiles, cols).save(out_path, quality=90)


def build_detail(samples, lut, name, max_edge, out_path):
    from PIL import Image
    cols = []
    for s in samples:
        t = _thumb(s, max_edge)
        g = Image.fromarray(apply_lut(np.asarray(t.convert("RGB")), lut, 1.0))
        pair = Image.new("RGB", (max(t.width, g.width), t.height + g.height + 6), (24, 24, 24))
        pair.paste(_label(t.copy(), "original"), (0, 0))
        pair.paste(_label(g, name), (0, t.height + 6))
        cols.append(pair)
    _montage(cols, len(cols)).save(out_path, quality=90)


def _open_file(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Batch (parallel)
# --------------------------------------------------------------------------- #
_W: Dict[str, Any] = {}


def _init_worker(lut, strength, quality, in_root, out_root, suffix):
    # Build the dense LUT once per worker; every image is then a fast gather.
    _W.update(dense=build_dense_lut(lut, strength), quality=quality,
              in_root=Path(in_root), out_root=Path(out_root), suffix=suffix)


def _process_one(in_path_str: str):
    from PIL import Image, ImageOps
    in_path = Path(in_path_str)
    try:
        img = Image.open(in_path)
        icc = img.info.get("icc_profile")
        # Bake the EXIF orientation into the pixels so the output displays the
        # right way up everywhere (exif_transpose also drops the now-applied
        # orientation tag, so it isn't rotated a second time by EXIF-aware apps).
        img = ImageOps.exif_transpose(img)
        exif = img.info.get("exif")
        graded = apply_dense(np.asarray(img.convert("RGB")), _W["dense"])
        rel = in_path.relative_to(_W["in_root"])
        out_path = _W["out_root"] / rel.parent / (rel.stem + _W["suffix"] + in_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save = {}
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            save = {"quality": _W["quality"], "subsampling": 0}
        if exif:
            save["exif"] = exif
        if icc:
            save["icc_profile"] = icc
        Image.fromarray(graded).save(out_path, **save)
        return (True, in_path.name, "")
    except Exception as exc:  # noqa: BLE001
        return (False, in_path.name, str(exc))


def run_batch(images, lut, cfg, in_root, out_root):
    from concurrent.futures import ProcessPoolExecutor
    workers = int(cfg["output"]["workers"]) or (os.cpu_count() or 4)
    strength = float(cfg["lut"]["strength"])
    quality = int(cfg["output"]["jpeg_quality"])
    suffix = str(cfg["output"]["suffix"])
    print(f"\nGrading {len(images)} image(s) with {workers} workers -> {out_root}")
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(images), desc="LUT")
    except Exception:
        bar = None
    done = failed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(lut, strength, quality, str(in_root), str(out_root), suffix)) as ex:
        for ok, name, err in ex.map(_process_one, [str(p) for p in images]):
            if ok:
                done += 1
            else:
                failed += 1
                print(f"[WARN] {name}: {err}")
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    print(f"--- done: {done} graded, {failed} failed -> {out_root} ---")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def list_luts(folder: Path) -> List[Path]:
    # Scan each file once and match the extension case-insensitively. (Globbing
    # *.cube AND *.CUBE double-counts every file on case-insensitive filesystems
    # like Windows, where both patterns match the same files.)
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".cube"),
                  key=lambda p: p.name.lower())


def list_images(folder: Path, recursive: bool) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def pick_samples(images: List[Path], k: int):
    from PIL import Image, ImageOps
    if not images:
        return []
    idx = np.linspace(0, len(images) - 1, min(k, len(images))).round().astype(int)
    out = []
    for i in sorted(set(idx.tolist())):
        try:
            out.append(ImageOps.exif_transpose(Image.open(images[i])).convert("RGB"))
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Runner LUT Suite (local color grading)")
    ap.add_argument("config", nargs="?", default="settings_lut.json")
    ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--luts")
    ap.add_argument("--lut", help="LUT filename or number to use (skips selection)")
    ap.add_argument("--yes", "-y", action="store_true", help="run the batch without confirmation")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open preview sheets")
    ap.add_argument("--recursive", action="store_true", help="include images in all subfolders (mirrors the tree to output)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg["paths"]["input_folder"] = args.input
    if args.output:
        cfg["paths"]["output_folder"] = args.output
    if args.recursive:
        cfg["paths"]["recursive"] = True
    if args.luts:
        cfg["paths"]["lut_folder"] = args.luts
    if args.workers is not None:
        cfg["output"]["workers"] = args.workers
    if args.no_open:
        cfg["options"]["auto_open_previews"] = False

    in_root = Path(cfg["paths"]["input_folder"])
    out_root = Path(cfg["paths"]["output_folder"])
    lut_dir = Path(cfg["paths"]["lut_folder"])
    prev_dir = Path(cfg["paths"]["preview_folder"] or (out_root / "_previews"))

    print("=" * 55)
    print("  Runner LUT Suite -- local color grading")
    print("=" * 55)

    luts = list_luts(lut_dir)
    if not luts:
        print(f"[ERROR] No .cube files in '{lut_dir}'. Drop your LUTs there and re-run.")
        sys.exit(1)
    # 'recursive' is accepted under paths (where the template shows it) or options.
    recursive = bool(cfg["paths"].get("recursive", cfg["options"].get("recursive", True)))
    images = list_images(in_root, recursive)
    if recursive:
        # don't re-ingest our own outputs / preview sheets if they live under the input tree
        skip = [out_root.resolve(), prev_dir.resolve()]
        images = [p for p in images if not any(s in p.resolve().parents for s in skip)]
    if not images:
        # Self-diagnose: report recursion state, what file types ARE present,
        # and whether there are unscanned subfolders.
        try:
            scan = list(in_root.rglob("*")) if in_root.exists() else []
        except Exception:
            scan = []
        exts = sorted({p.suffix.lower() for p in scan if p.is_file() and p.suffix})
        subdirs = [p.name for p in in_root.iterdir() if p.is_dir()] if in_root.exists() else []
        print(f"[ERROR] No supported images in '{in_root}' (recursive={recursive}).")
        print(f"        Supported: {', '.join(sorted(IMG_EXTS))}")
        if exts:
            print(f"        File types found in the tree: {', '.join(exts)}")
            raw = [e for e in exts if e in (".nef", ".arw", ".cr2", ".cr3", ".raf", ".rw2", ".dng", ".orf")]
            if raw:
                print(f"        {', '.join(raw)} are RAW files -- not supported. Export them to JPEG/TIFF first.")
        if subdirs and not recursive:
            print(f"        {len(subdirs)} subfolder(s) present but recursive is OFF -- set paths.recursive=true or pass --recursive.")
        sys.exit(1)
    print(f"Found {len(luts)} LUT(s) and {len(images)} image(s)"
          + (" across all subfolders" if recursive else "") + ".")

    # ---- resolve a pre-selected LUT (config or --lut) ----
    chosen_name = args.lut or cfg["lut"]["selected"]
    chosen: Optional[Path] = None
    if chosen_name:
        if str(chosen_name).isdigit():
            i = int(chosen_name) - 1
            if 0 <= i < len(luts):
                chosen = luts[i]
        else:
            for p in luts:
                if p.name.lower() == str(chosen_name).lower():
                    chosen = p
                    break
        if chosen is None:
            print(f"[WARN] LUT '{chosen_name}' not found; falling back to interactive selection.")

    # ---- STAGE 1: preview / select ----
    confirmed = False
    if chosen is None:
        prev_dir.mkdir(parents=True, exist_ok=True)
        parsed = []
        names = []
        for p in luts:
            try:
                parsed.append(parse_cube(p))
                names.append(p.stem)
            except Exception as exc:
                print(f"[WARN] skipping {p.name}: {exc}")
        if not parsed:
            print("[ERROR] No valid LUTs could be parsed.")
            sys.exit(1)
        samples = pick_samples(images, int(cfg["output"]["sample_count"]))
        max_edge = int(cfg["output"]["preview_max_edge"])
        overview = prev_dir / "_overview_all_luts.jpg"
        print("\nRendering overview of all LUTs ...")
        build_overview(samples[0], parsed, names, max_edge, overview)
        print(f"  -> {overview}")
        if cfg["options"]["auto_open_previews"]:
            _open_file(overview)

        valid = [p for p in luts if p.stem in names]
        while True:
            print("\nAvailable LUTs:")
            for i, nm in enumerate(names, 1):
                print(f"  [{i}] {nm}")
            sel = input("\nSelect a LUT number to preview (or 'q' to quit): ").strip().lower()
            if sel in ("q", "quit", "exit"):
                print("Cancelled.")
                return
            if not sel.isdigit() or not (1 <= int(sel) <= len(valid)):
                print("  Please enter a valid number.")
                continue
            k = int(sel) - 1
            detail = prev_dir / f"preview_{names[k]}.jpg"
            build_detail(samples, parsed[k], names[k], max_edge, detail)
            print(f"  Preview of '{names[k]}' on {len(samples)} sample(s) -> {detail}")
            if cfg["options"]["auto_open_previews"]:
                _open_file(detail)
            ans = input(f"Apply '{names[k]}' to ALL {len(images)} images? [y]es / [n]o pick another / [q]uit: ").strip().lower()
            if ans in ("y", "yes"):
                chosen = valid[k]
                confirmed = True   # "apply to all" already confirms the batch
                break
            if ans in ("q", "quit", "exit"):
                print("Cancelled.")
                return
            # else loop and pick another

    # ---- STAGE 2: batch ----
    lut = parse_cube(chosen)
    print(f"\nSelected LUT: {chosen.name}"
          + (f"  (\"{lut['title']}\")" if lut.get("title") else "")
          + f"   strength={cfg['lut']['strength']}")
    if not args.yes and not confirmed:
        ans = input(f"Grade all {len(images)} images -> {out_root} ? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return
    out_root.mkdir(parents=True, exist_ok=True)
    run_batch(images, lut, cfg, in_root, out_root)


if __name__ == "__main__":
    main()
