# SISR
_A Pre-trained Single-Image Super-Resolution Toolbox for Mars and Earth Observations_

**Author**: Dr. Yu Tao @ Surrey AI Imaging Limited  
**Core architecture**: SwinIR (acknowledgement: https://github.com/JingyunLiang/SwinIR)

This repository contains research-oriented code for single-image super-resolution (SISR) of:
- Terrestrial EO thermal infrared (TIR/MWIR) and visible/RGB imagery
- Planetary Mars imagery (HiRISE/CTX/HRSC/CaSSIS) — work-in-progress
- Environmental single-band rasters (e.g., PM2.5, albedo; UInt16/Float32)
- Ultra‑high‑resolution (UHR) RGB aerial/satellite data

The pipeline is designed around 4× upscaling with geospatial metadata preserved end‑to‑end (GeoTIFF ⇄ PNG ⇄ GeoTIFF).

---

## Table of Contents
1. [What’s Included](#whats-included)
2. [Requirements](#requirements)
3. [Pre-trained Weights](#pre-trained-weights)
4. [Quick Start (3-step pipeline)](#quick-start-3-step-pipeline)
5. [One‑command Automation Scripts](#one-command-automation-scripts)
6. [Script Reference](#script-reference)
7. [Examples](#examples)
8. [Model/Dataset Compatibility](#modeldataset-compatibility)
9. [Troubleshooting](#troubleshooting)
10. [Acknowledgements & Citation](#acknowledgements--citation)
11. [License](#license)

---

## What’s Included

```
prep_geotiff_input.py      # GeoTIFF → header(.x4header.tif) + PNG (1-band scaled to 1..255; 3-band unchanged)
prep_geotiff_output.py     # PNG + x4header + original GeoTIFF → super-resolved GeoTIFF (COG optional)
inference.py               # SwinIR inference (auto small/large handling with tiling)
sisr_auto_large.py         # Generic tiling/mosaic pipeline for very large images
sisr_auto_landsat_tir_single_band.py  # Landsat-like single-band (UInt16/Float32)
sisr_auto_s2albedo_single_band.py     # Sentinel‑2 albedo single-band (UInt16/Float32)
sisr_auto_pm25_single_band.py         # PM2.5 single-band (UInt16/Float32, experimental)
sisr_auto_sentinel-2_rgb.py           # Sentinel‑2 RGB (3-band, 8-bit) + optional COG + CLEAN
sisr_auto_uhr_rgb.py                  # UHR RGB (3-band, 8-bit) + optional COG + CLEAN
network_architecture.py    # Model definition utilities (SwinIR-based)
```

---

## Requirements

- Python 3.8+
- GDAL (with Python bindings)
- OpenCV (`opencv-python`)
- NumPy
- PyTorch (CUDA recommended)
- (Optional) A GPU with sufficient memory for faster inference

> Install system GDAL first (varies by OS). Then:
```bash
pip install numpy opencv-python torch torchvision
# Ensure GDAL’s Python bindings match your system GDAL
```

---

## Pre-trained Weights

Download pre-trained weights (m‑* families) from Google Drive:

- https://drive.google.com/drive/folders/1KHGWjFf1ZkvSvsjYT-mCSusjUJpetQSW?usp=drive_link

> Place weight files under `pre-trained-models/` or pass the absolute path via `-m` / `--model_path`.

---

## Quick Start (3-step pipeline)

**Goal**: take a GeoTIFF → super-resolve with SwinIR → write a georeferenced GeoTIFF.

1. **Prepare input** (creates an upsampled header `.x4header.tif` and PNG):
   ```bash
   python prep_geotiff_input.py <input.tif> <work_dir>
   # Produces: <work_dir>/<name>.png  and  <work_dir>/<name>.x4header.tif
   ```

2. **Run inference** (4× SR, auto‑tiling for large inputs):
   ```bash
   python inference.py -m <path/to/weights.pth> -i <work_dir>/<name>.png -o <work_dir>/<name>.srr.png
   ```

3. **Write GeoTIFF** (restore original data type, nodata, georeferencing; optional COG):
   ```bash
   python prep_geotiff_output.py <work_dir>/<name>.srr.png <work_dir>/<name>.x4header.tif <out>/<name>.srr.tif <input.tif> [-COG]
   ```

---

## One-command Automation Scripts

These wrappers split large rasters into overlapping tiles, run inference, and mosaic the results. They also validate model/data compatibility and can optionally emit a Cloud‑Optimized GeoTIFF (`-COG`) and delete intermediates (`-CLEAN`).

### Sentinel‑2 RGB (3‑band, 8‑bit)
```bash
python sisr_auto_sentinel-2_rgb.py <input(.tif|.jp2)> <out_dir> <weights> [-COG] [-CLEAN]
# expects weights starting with m-1
```

### UHR RGB (3‑band, 8‑bit)
```bash
python sisr_auto_uhr_rgb.py <input(.tif|.jp2)> <out_dir> <weights> [-COG] [-CLEAN]
# expects weights starting with m-5
```

### Landsat TIR single‑band (UInt16/Float32)
```bash
python sisr_auto_landsat_tir_single_band.py <input(.tif|.jp2)> <out_dir> <weights> [-COG] [-CLEAN]
# expects weights starting with m-2
```

### Sentinel‑2 Albedo single‑band (UInt16/Float32)
```bash
python sisr_auto_s2albedo_single_band.py <input(.tif|.jp2)> <out_dir> <weights> [-COG] [-CLEAN]
# expects weights starting with m-4
```

### PM2.5 single‑band (UInt16/Float32, experimental)
```bash
python sisr_auto_pm25_single_band.py <input(.tif|.jp2)> <out_dir> <weights> [-COG] [-CLEAN]
# expects weights starting with m-2 or m-4
```

### Generic large-image pipeline
```bash
python sisr_auto_large.py <input(.tif|.jp2)> <out_dir> <weights>
# 3-band Byte → m-1 or m-5; 1-band UInt16/Float32 → m-2
```

---

## Script Reference

### `prep_geotiff_input.py`
- **Usage**: `python prep_geotiff_input.py <input> <output_dir>`  
- **Behaviour**:  
  - Validates GeoTIFF (requires a non‑identity geotransform).  
  - Writes `<name>.x4header.tif` by 4× upsampling geotransform (nearest neighbour).  
  - Writes `<name>.png`:
    - 1‑band: linearly scales original min/max → 1..255 (Byte); does **not** set nodata in PNG.
    - 3‑band: no scaling; preserves visual range.
- **Notes**: header preserves original nodata and projection for later restoration.

### `inference.py`
- **Usage**: `python inference.py -m <weights.pth> -i <input.png> -o <output.srr.png>`  
- **Behaviour**: loads SwinIR weights, auto‑detects small vs large inputs (tile size=512), reuses GPU where available, accumulates on CPU to reduce VRAM; writes an 8‑bit PNG.  
- **Channels**: if the input PNG is 1‑band, output is converted back to single‑band.

### `prep_geotiff_output.py`
- **Usage**: `python prep_geotiff_output.py <sr_png> <x4header.tif> <out.tif> <original_input.tif> [-COG]`  
- **Behaviour**: rebuilds a GeoTIFF in the original data type and CRS; 1‑band products are reverse‑scaled to the original dynamic range using the original min/max and nodata; optional COG output.

### Automation scripts (all support JP2→GeoTIFF conversion automatically)
Common features:
- Tile size 2000 px with 200 px overlap (mosaic after SR)
- Build header+PNG per tile; run `inference.py`; write per‑tile GeoTIFF and mosaic
- `-COG` to emit Cloud‑Optimized GeoTIFF; `-CLEAN` to delete intermediates

Per‑script constraints:
- `sisr_auto_sentinel-2_rgb.py`: input must be 3‑band 8‑bit; weights `m-1*`  
- `sisr_auto_uhr_rgb.py`: input must be 3‑band 8‑bit; weights `m-5*`  
- `sisr_auto_landsat_tir_single_band.py`: input must be 1‑band UInt16/Float32; weights `m-2*`  
- `sisr_auto_s2albedo_single_band.py`: input must be 1‑band UInt16/Float32; weights `m-4*`  
- `sisr_auto_pm25_single_band.py`: input must be 1‑band UInt16/Float32; weights `m-2*` or `m-4*` (experimental)

---

## Examples

### Minimal 3‑step example
```bash
# paths
IMG=/data/landsat/LST_206024_20230614.tif
TMP=/data/landsat/tmp
OUT=/data/landsat
W=pre-trained-models/m-2_psnr.pth

# 1) prepare
python prep_geotiff_input.py $IMG $TMP

# 2) inference
python inference.py -m $W -i $TMP/LST_206024_20230614.png -o $TMP/LST_206024_20230614.srr.png

# 3) GeoTIFF
python prep_geotiff_output.py $TMP/LST_206024_20230614.srr.png $TMP/LST_206024_20230614.x4header.tif $OUT/LST_206024_20230614.srr.tif $IMG
```

### One‑liner (Sentinel‑2 RGB → COG, clean intermediates)
```bash
python sisr_auto_sentinel-2_rgb.py S2_RGB.jp2 out_dir pre-trained-models/m-1_xxx.pth -COG -CLEAN
```

### UHR RGB aerial
```bash
python sisr_auto_uhr_rgb.py city_rgb.tif out_dir pre-trained-models/m-5_xxx.pth -COG
```

### Landsat TIR single‑band
```bash
python sisr_auto_landsat_tir_single_band.py L8_TIR.tif out_dir pre-trained-models/m-2_xxx.pth -COG
```

### Sentinel‑2 albedo (single‑band)
```bash
python sisr_auto_s2albedo_single_band.py s2_albedo.tif out_dir pre-trained-models/m-4_xxx.pth -COG
```

### PM2.5 single‑band (experimental)
```bash
python sisr_auto_pm25_single_band.py pm25.tif out_dir pre-trained-models/m-4_xxx.pth -COG -CLEAN
```

---

## Model/Dataset Compatibility

| Script / Data                             | Bands | DType           | Expected weights prefix |
|-------------------------------------------|:-----:|-----------------|-------------------------|
| Sentinel‑2 RGB                            | 3     | Byte (8‑bit)    | `m-1*`                  |
| UHR RGB                                   | 3     | Byte (8‑bit)    | `m-5*`                  |
| Landsat TIR single‑band                   | 1     | UInt16/Float32  | `m-2*`                  |
| Sentinel‑2 albedo single‑band             | 1     | UInt16/Float32  | `m-4*`                  |
| PM2.5 single‑band (experimental)          | 1     | UInt16/Float32  | `m-2*` or `m-4*`        |
| Generic large-image (`sisr_auto_large.py`) | 1/3   | see above       | 3‑band→`m-1`/`m-5`; 1‑band→`m-2` |

> All pipelines use 4× scaling. PNG intermediates are in 8‑bit; single‑band data are reverse‑scaled back to original units/types in the final GeoTIFF, with nodata restored.

---

## Troubleshooting

- **“expected 3‑band 8‑bit RGB imagery”**: check your input has 3 bands and Byte data type. Convert if needed.
- **“expected single‑band UInt16 or Float32”**: cast/convert your raster (e.g., `gdal_translate -ot UInt16`).
- **Seams or halos near nodata**: scripts dilate the nodata mask; PM2.5 pipeline additionally replaces a 4‑px rim from the header to suppress halos.
- **Very large images**: the wrappers tile at 2000 px (200 px overlap). `inference.py` auto‑tiles internally for big inputs (tile=512).
- **Cloud‑Optimized GeoTIFF**: add `-COG` to supported scripts or to `prep_geotiff_output.py`.
- **Disk usage**: add `-CLEAN` to delete temporary PNG/headers and per‑tile results.
- **JP2 inputs**: wrappers convert `.jp2` to GeoTIFF automatically.

---

## Acknowledgements & Citation

- This work uses and acknowledges the **SwinIR** architecture: https://github.com/JingyunLiang/SwinIR  
- If you find this repo useful in research, please cite the repo and SwinIR accordingly.

```
@software{tao_sisr_toolbox,
  author  = {Yu Tao},
  title   = {SISR: A Pre-trained Single-Image Super-Resolution Toolbox for Mars and Earth Observations},
  year    = {2025},
  note    = {Surrey AI Imaging Limited},
  url     = {https://github.com/yutao3/SISR}
}
```

---

## License

Research-only usage. Contact **Surrey AI Imaging Limited** for commercial licensing inquiries.

