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

## Make a LUT from a reference image
Have a shot whose look you love? Extract a `.cube` from it — the tool analyses
its white balance, tonal range (lifted/crushed blacks), gamma, saturation, and
shadow/highlight split-tone and bakes them into a LUT:
```
.\.venv\Scripts\python.exe make_lut_from_image.py "D:/hero.jpg"
.\.venv\Scripts\python.exe make_lut_from_image.py "D:/hero.jpg" luts/MyLook.cube --strength 0.8
```
It writes to `luts/<image>.cube` by default, so it shows up in the picker on the
next `run_lut.bat`. `--strength` (0..1) dials the intensity. Note it extracts a
consistent *grade*, not a pixel-perfect copy — a flat/hazy source yields a
flat/hazy LUT, so point it at a shot with the treatment you actually want.

## Make a LUT from a before/after pair (most accurate)
If you have the **same photo un-graded and graded** (e.g. the original export and
your edited version), this reproduces the *exact* edit — exposure, white balance,
tone curve, saturation, HSL, split-tone — by fitting the colour transform between
them. No XMP needed; the grade is already baked into the "after" pixels.
```
.\.venv\Scripts\python.exe make_lut_from_pair.py "D:/original.jpg" "D:/graded.jpg"
.\.venv\Scripts\python.exe make_lut_from_pair.py "D:/original.jpg" "D:/graded.jpg" luts/MyGrade.cube --degree 3
```
The two must be the **same framing** (only colour differs — no crop/rotate). It
prints a fit RMSE so you know how faithfully it captured the edit. Prefer this
over `make_lut_from_image.py` whenever you have the before/after pair.

## Make a LUT from a Lightroom / XMP edit (Hald CLUT — best for presets)
The professional way to turn a Lightroom develop look (your XMP edits) into a
`.cube`. It captures the *entire* pipeline (WB, tone curve, HSL, colour grading,
calibration) exactly, across the whole colour range — better than either method
above:
```
.\.venv\Scripts\python.exe make_hald.py            # -> hald_identity.png
```
1. In **Lightroom**: import `hald_identity.png`, copy the **Develop settings**
   from an edited photo, and **Paste** them onto the Hald. Turn OFF crop,
   geometry, sharpening, noise reduction, and vignette — **colour/tone only**.
2. **Export** it as PNG or TIFF: **sRGB**, **no sharpening**, **no resize**.
3. Convert it:
```
.\.venv\Scripts\python.exe hald_to_cube.py "hald_MyLook.png" luts/MyLook.cube
```
Repeat step 1–3 (paste a different look each time) to bank several LUTs. Verified
round-trip fidelity ~0.6/255.

## Check your LUTs are different
Confirm two exported `.cube`s really are distinct grades (not the same look
twice), and see what each does (WB, black lift, contrast, saturation):
```
.\.venv\Scripts\python.exe compare_luts.py luts/Look1.cube luts/Look2.cube
.\.venv\Scripts\python.exe compare_luts.py            # compares every LUT in luts/
```
It prints a one-line character for each and a pairwise diff; `IDENTICAL LUTs`
means two files are the same grade.

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
