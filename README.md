# Runner LUT Suite

A small **local** tool that applies a color-grade LUT to a whole folder of
images. Runs entirely on **CPU** — no GPU, no torch. Meant to be the *final
look* pass after your crops are cropped/upscaled/face-restored.

Two stages:

1. **Preview / select** — applies every `.cube` in `luts/` to a few sample
   images, writes side-by-side preview sheets, opens them, and lets you pick one
   interactively. Reject and try another until you're happy.
2. **Batch** — applies the chosen LUT to the entire input folder, in parallel
   across all CPU cores.

## Setup (Windows)
```
winget install -e --id Python.Python.3.12    (if Python isn't installed)
setup_lut.bat
```
`setup_lut.bat` builds a venv, installs `numpy` / `Pillow` / `tqdm`, and writes
six **starter LUTs** into `luts\` so it works out of the box.

## Use
1. Put your own `.cube` files into `luts\` (they sit alongside the starters).
2. Edit `settings_lut.json` → `input_folder` and `output_folder`.
3. `run_lut.bat`
4. When the overview opens, pick a LUT number, check the preview, and confirm.
   It then grades the whole folder.

## Adding your own LUTs
Drop any standard Adobe `.cube` file (1D or 3D) into `luts\`. That's it — the
suite lists whatever it finds. The included `make_sample_luts.py` shows how the
starters were generated if you want to build your own programmatically.

## Command-line (optional)
```
python lut_suite.py --input D:/crops --output D:/graded            # interactive
python lut_suite.py --lut WarmFilm.cube --yes                      # skip prompts, grade now
python lut_suite.py --input D:/crops --output D:/graded --workers 16
```

## Config quick reference
| key | meaning |
|---|---|
| `input_folder` / `output_folder` | where images come from / go to |
| `lut_folder` | folder scanned for `.cube` files (default `luts`) |
| `lut.selected` | filename to skip the picker (e.g. `"WarmFilm.cube"`); `null` = ask |
| `lut.strength` | 0..1 blend of the graded look over the original (1.0 = full) |
| `output.jpeg_quality` | JPEG quality for `.jpg`/`.jpeg` outputs |
| `output.workers` | parallel CPU workers (`0` = all cores) |
| `output.suffix` | appended to output filenames (e.g. `"_graded"`) |
| `output.sample_count` | how many images to preview across |
| `options.recursive` | also grade images in subfolders (mirrors the tree) |

## Notes
- **Format & metadata preserved** — `.jpg` stays `.jpg`, `.png` stays `.png`;
  EXIF and ICC profiles are carried through.
- **No upscaling / no face work** — this is *only* the color grade. Do the
  enhancement upstream, then run this last.
- **Fast** — it pre-expands the chosen LUT into a dense table once, so each
  image is a single lookup. On a many-core laptop, ~100k images is roughly
  10–20 minutes (dominated by JPEG decode/encode, not the LUT).
