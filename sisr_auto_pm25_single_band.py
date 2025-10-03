# sisr_auto_pm25_single_band.py
# -------------------------------------------------------------
# Additional Super‑resolution pipeline for single‑band PM2.5 map
# with per‑tile dynamic range stretching (1‑255), automatic
# reverse‑scaling, mean/σ harmonisation against the input geoheader,
# and nodata safety check for UInt16 and Float32 data.
# -------------------------------------------------------------

from __future__ import annotations  # tuple[list] et al. work on 3.7-3.8; harmless ≥3.9

import sys
import os
import subprocess
import time
import shutil
import numpy as np
import cv2
from osgeo import gdal, gdal_array, osr

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def get_geo_info(ds):
    """Return a dict with basic GDAL geoinfo about *ds*."""
    nod = ds.GetRasterBand(1).GetNoDataValue()
    return {
        "nodata": 0 if nod is None else nod,
        "gt": ds.GetGeoTransform(),
        "proj": ds.GetProjection(),
        "dtype": ds.GetRasterBand(1).DataType,
        "size": (ds.RasterXSize, ds.RasterYSize),
    }


def expand_nodata_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate *mask* (bool) so inferred edges are eroded one pixel inwards."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)


# -----------------------------------------------------------------------------
# 1.  Stretch tile to 1‑255 (nodata preserved) before inference
# -----------------------------------------------------------------------------

def stretch_to_byte(src_tif: str, dst_png: str) -> tuple[float, float]:
    """Translate *src_tif* → 8‑bit *dst_png* with values 1‑255.

    Returns
    -------
    tuple
        (min_original, max_original) collected from valid pixels only.
    """
    ds = gdal.Open(src_tif)
    rb = ds.GetRasterBand(1)
    nod = rb.GetNoDataValue()
    arr = rb.ReadAsArray()

    # Exclude nodata from statistics
    valid = arr != (0 if nod is None else nod)
    mn = float(arr[valid].min())
    mx = float(arr[valid].max())

    # Write PNG for network input
    gdal.Translate(
        dst_png,
        ds,
        options=gdal.TranslateOptions(
            format="PNG",
            outputType=gdal.GDT_Byte,
            scaleParams=[[mn, mx, 1, 255]],
        ),
    )
    ds = None
    return mn, mx


# -----------------------------------------------------------------------------
# 2.  Reverse stretch and harmonise mean/std w.r.t. x4 header
# -----------------------------------------------------------------------------

def _match_mean_std(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return *src* scaled so mean/σ match *ref* inside *mask*."""
    src_vals = src[mask].astype(np.float64)
    ref_vals = ref[mask].astype(np.float64)

    if src_vals.size == 0:
        # pathological tile (fully nodata) – skip
        return src

    mean_s, std_s = src_vals.mean(), src_vals.std()
    mean_r, std_r = ref_vals.mean(), ref_vals.std()

    if std_s < 1e-6:
        return src  # avoid divide‑by‑zero, tile is flat anyway

    scale = std_r / std_s
    out = (src.astype(np.float64) - mean_s) * scale + mean_r
    return out


# -----------------------------------------------------------------------------
# 3.  End‑to‑end helper: PNG → GeoTIFF with reverse scaling & harmonisation
# -----------------------------------------------------------------------------

"""
def png_to_geotiff(srr_png: str, header_tif: str, out_tif: str, original_min: float, original_max: float):
    # Write the SRR *srr_png* to *out_tif* with original dynamic range and mean/σ harmonised against *header_tif*.
    # ---- load SRR PNG (network result)
    img = cv2.imread(srr_png, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hdr_ds = gdal.Open(header_tif)
    info = get_geo_info(hdr_ds)
    nod = info["nodata"]
    hdr_arr = hdr_ds.ReadAsArray()
    hdr_ds = None

    # Build mask (dilated) on header nodata
    mask_nodata = hdr_arr == nod
    mask_valid = ~expand_nodata_mask(mask_nodata, iterations=1)

    # ---- reverse linear stretch 1‑255 → original units
    srr_arr = img.astype(np.float64)
    srr_arr = original_min + (srr_arr - 1.0) * (original_max - original_min) / 254.0

    # ---- mean/σ harmonisation (only on valid data)
    srr_arr_harmonised = _match_mean_std(srr_arr, hdr_arr, mask_valid)

    # reinstate nodata exactly
    srr_arr_harmonised[mask_nodata] = nod

    # ---- cast to original dtype & write GeoTIFF
    np_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(info["dtype"])
    out_arr = srr_arr_harmonised.astype(np_dtype)

    drv = gdal.GetDriverByName("GTiff")
    xsize, ysize = info["size"]
    ds_out = drv.Create(out_tif, xsize, ysize, 1, info["dtype"])
    ds_out.SetGeoTransform(info["gt"])
    ds_out.SetProjection(info["proj"])
    rb = ds_out.GetRasterBand(1)
    rb.WriteArray(out_arr)
    rb.SetNoDataValue(nod)
    ds_out = None
"""

# -----------------------------------------------------------------------------
# 3.  End-to-end helper: PNG → GeoTIFF with reverse-scaling, harmonisation
#     …and now a 4-pixel “halo” replacement taken from the x4-header
# -----------------------------------------------------------------------------
def png_to_geotiff(
        srr_png: str,
        header_tif: str,
        out_tif: str,
        original_min: float,
        original_max: float
    ):
    """
    Convert the network PNG result to GeoTIFF:

    1. Reverse the 1-255 stretch back to the original dynamic range
    2. Match mean / std to the x4-header (valid data only)
    3. *New*: replace the first 4-pixel rim around nodata with the
       corresponding pixels from the x4-header to hide edge artefacts
    """
    # ---- load SRR PNG
    img = cv2.imread(srr_png, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:                        # safety: strip possible RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- load x4 header & geo-info
    hdr_ds  = gdal.Open(header_tif)
    info    = get_geo_info(hdr_ds)
    nod     = info["nodata"]
    hdr_arr = hdr_ds.ReadAsArray()
    hdr_ds  = None

    # ---- masks
    mask_nodata = hdr_arr == nod
    mask_valid  = ~expand_nodata_mask(mask_nodata, iterations=1)   # for mean/σ
    mask_border = (expand_nodata_mask(mask_nodata, iterations=4)   # 4-px rim
                   & ~mask_nodata)

    # ---- reverse 8-bit stretch → original units (float64 for accuracy)
    srr_arr = img.astype(np.float64)
    srr_arr = original_min + (srr_arr - 1.0) * (original_max - original_min) / 254.0

    # ---- mean / σ harmonisation on valid data
    srr_arr = _match_mean_std(srr_arr, hdr_arr, mask_valid)

    # ---- NEW: copy 4-pixel rim from header to suppress halo artefacts
    srr_arr[mask_border] = hdr_arr[mask_border]

    # ---- reinstate nodata exactly
    srr_arr[mask_nodata] = nod

    # ---- cast to original dtype & write GeoTIFF
    np_dtype  = gdal_array.GDALTypeCodeToNumericTypeCode(info["dtype"])
    out_arr   = srr_arr.astype(np_dtype)

    drv       = gdal.GetDriverByName("GTiff")
    xsize, ysize = info["size"]
    ds_out    = drv.Create(out_tif, xsize, ysize, 1, info["dtype"])
    ds_out.SetGeoTransform(info["gt"])
    ds_out.SetProjection(info["proj"])
    rb        = ds_out.GetRasterBand(1)
    rb.WriteArray(out_arr)
    rb.SetNoDataValue(nod)
    ds_out    = None



# -----------------------------------------------------------------------------
# 4.  Pre‑processing: convert JP2 → GeoTIFF & (optionally) tile
# -----------------------------------------------------------------------------

def convert_jp2_to_tif(input_path: str, output_dir: str) -> str:
    print("Step 0: Converting JP2 to GeoTIFF…")
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_tif = os.path.join(output_dir, base + ".tif")
    if gdal.Translate(out_tif, input_path, options=gdal.TranslateOptions(format="GTiff")) is None:
        raise RuntimeError("Failed to convert JP2 to GeoTIFF.")
    print(f"GeoTIFF saved to: {out_tif}")
    return out_tif


def tile_image(input_tif: str, output_dir: str, tile_size: int = 2000, overlap: int = 200):
    print("Step 1: Splitting input into tiles…")
    ds = gdal.Open(input_tif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    step = tile_size - overlap
    base = os.path.splitext(os.path.basename(input_tif))[0]
    tiles = []
    for y in range(0, ysize, step):
        for x in range(0, xsize, step):
            w = min(tile_size, xsize - x)
            h = min(tile_size, ysize - y)
            out = os.path.join(output_dir, f"{base}_tile_{y}_{x}.tif")
            gdal.Translate(out, ds, options=gdal.TranslateOptions(srcWin=[x, y, w, h], format="GTiff")).FlushCache()
            tiles.append(out)
    ds = None
    print(f"Created {len(tiles)} tiles.")
    return tiles


def mosaic_tiles(tile_paths: list[str], output_path: str):
    print("Step 4: Merging tiles into final mosaic…")
    gdal.Warp(output_path, tile_paths, options=gdal.WarpOptions(format="GTiff"))
    print(f"Final mosaic written to: {output_path}")

# -----------------------------------------------------------------------------
# 5.  Per‑tile SRR processing
# -----------------------------------------------------------------------------

def prepare_inputs(tile_tif: str, work_dir: str):
    """Create x4 header GeoTIFF & stretched PNG for *tile_tif*."""
    base = os.path.splitext(os.path.basename(tile_tif))[0]

    # 5.1 Upsampled x4 header (GeoTIFF, same dtype as input)
    ds = gdal.Open(tile_tif)
    gt = ds.GetGeoTransform()
    hdr_path = os.path.join(work_dir, base + ".x4header.tif")
    gdal.Translate(
        hdr_path,
        ds,
        options=gdal.TranslateOptions(
            xRes=gt[1] / 4.0,
            yRes=gt[5] / 4.0,
            resampleAlg="near",
            format="GTiff",
        ),
    )

    # 5.2 8‑bit PNG + collect per‑tile min/max
    png_path = os.path.join(work_dir, base + ".png")
    orig_min, orig_max = stretch_to_byte(tile_tif, png_path)

    ds = None
    return png_path, hdr_path, orig_min, orig_max


def run_inference(png_in: str, weights: str, work_dir: str):
    base = os.path.splitext(os.path.basename(png_in))[0]
    srr_png = os.path.join(work_dir, base + ".srr.png")
    start = time.time()
    res = subprocess.run(["python", "inference.py", "-m", weights, "-i", png_in, "-o", srr_png], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    print(f"Inference completed in {time.time() - start:.1f} s → {srr_png}")
    return srr_png, time.time() - start


def process_tile(tile_tif: str, out_dir: str, weights: str):
    """Full pipeline for one *tile_tif*. Returns (final_tif_path, tmp_paths, dt_infer)."""
    base = os.path.splitext(os.path.basename(tile_tif))[0]

    # 1️⃣ Prepare inputs
    png_in, hdr_tif, mn, mx = prepare_inputs(tile_tif, out_dir)

    # 2️⃣ Neural inference
    srr_png, dt_inf = run_inference(png_in, weights, out_dir)

    # 3️⃣ Convert PNG → GeoTIFF with reverse scaling & harmonisation
    final_tif = os.path.join(out_dir, base + ".srr.tif")
    png_to_geotiff(srr_png, hdr_tif, final_tif, mn, mx)

    tmp_files = [png_in, hdr_tif, srr_png]
    return final_tif, tmp_files, dt_inf


# -----------------------------------------------------------------------------
# 6.  House‑keeping helpers
# -----------------------------------------------------------------------------

def safe_delete(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"Warning: could not delete {p}: {e}")

# -----------------------------------------------------------------------------
# 7.  Main entry point
# -----------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    cog = "-COG" in args
    clean = "-CLEAN" in args
    args = [a for a in args if a not in ("-COG", "-CLEAN")]

    if len(args) != 3:
        print("Usage: python sisr_auto_pm25_single_band_v2.py <input> <out_dir> <weights> [-COG] [-CLEAN]")
        sys.exit(1)

    inp_path, out_dir, weights = args

    t0 = time.time()

    print("### PM2.5 Single‑Band Super‑Resolution (experimental) ###")

    if not os.path.isfile(inp_path):
        print("Error: input file not found"); sys.exit(1)
    if not os.path.isfile(weights):
        print("Error: weights file not found"); sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    # Sanity check on model
    base_weights = os.path.splitext(os.path.basename(weights))[0]
    if not base_weights.startswith(("m-2", "m-4")):
        print("Model mismatch for M‑2/M-4: expected weights compatible to similar dataset.")
        sys.exit(1)

    # JP2 → GeoTIFF if needed
    if inp_path.lower().endswith(".jp2"):
        inp_path = convert_jp2_to_tif(inp_path, out_dir)

    ds_in = gdal.Open(inp_path)
    dtype_name = gdal.GetDataTypeName(ds_in.GetRasterBand(1).DataType)
    if ds_in.RasterCount != 1 or dtype_name not in ("UInt16", "Float32"):
        print("Error: expected single‑band UInt16 or Float32 raster."); sys.exit(1)
    xsz, ysz = ds_in.RasterXSize, ds_in.RasterYSize
    ds_in = None

    inference_total = 0.0
    temp_files: list[str] = []
    sr_tiles: list[str] = []
    orig_tiles: list[str] = []

    # Choose tiling threshold as before (>3k px)
    if max(xsz, ysz) > 3000:
        tiles = tile_image(inp_path, out_dir)
        for idx, tile in enumerate(tiles, 1):
            print(f"\n— Processing tile {idx}/{len(tiles)} —")
            out_tif, tmp, dt_inf = process_tile(tile, out_dir, weights)
            sr_tiles.append(out_tif)
            orig_tiles.append(tile)
            temp_files.extend(tmp)
            inference_total += dt_inf

        mosaic_path = os.path.join(out_dir, os.path.splitext(os.path.basename(inp_path))[0] + ".srr.tif")
        mosaic_tiles(sr_tiles, mosaic_path)
        final_output = mosaic_path
        temp_files.extend(sr_tiles)
        temp_files.extend(orig_tiles)
    else:
        final_output, tmp, dt_inf = process_tile(inp_path, out_dir, weights)
        temp_files.extend(tmp)
        inference_total += dt_inf

    if cog:
        print("Converting to Cloud‑Optimised GeoTIFF…")
        cog_out = os.path.splitext(final_output)[0] + ".cog.tif"
        gdal.Translate(cog_out, final_output, options=gdal.TranslateOptions(format="COG"))
        final_output = cog_out

    if clean:
        print("Cleaning up intermediate files…")
        safe_delete(temp_files)

    print("\nPipeline finished.")
    print(f"Total inference time : {inference_total:.1f} s")
    print(f"Overall wall time    : {time.time() - t0:.1f} s")
    print(f"Final output         : {final_output}")


if __name__ == "__main__":
    main()

