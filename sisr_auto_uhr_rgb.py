#!/usr/bin/env python3
"""
Ultra-High-Resolution (UHR) RGB image super‑resolution pipeline
-----------------------------------------
Usage:
    python sisr_auto_uhr_rgb.py <input> <out_dir> <weights> [-COG] [-CLEAN]

The script still relies on *inference.py* for the actual deep learning based inference step
and on GDAL for geospatial I/O.
"""

import sys
import os
import subprocess
import time
from typing import List
import cv2
from osgeo import gdal

gdal.UseExceptions()  # let GDAL raise Python exceptions on errors

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Uniform log prefix."""
    print(f"[INFO] {msg}")


def _delete_files(files: List[str]) -> None:
    """Silently delete a list of files if they exist."""
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            # We do not want cleaning errors to stop the script
            pass


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def convert_jp2_to_tif(input_path: str, output_dir: str) -> str:
    _log("Converting JP2 input to GeoTIFF …")
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_tif = os.path.join(output_dir, f"{base}.tif")
    opts = gdal.TranslateOptions(format="GTiff")
    if gdal.Translate(out_tif, input_path, options=opts) is None:
        raise RuntimeError("Failed to convert JP2 to GeoTIFF")
    _log(f"GeoTIFF saved: {out_tif}")
    return out_tif


def tile_image(input_tif: str, output_dir: str, tile_size: int = 2000, overlap: int = 200) -> List[str]:
    _log("Splitting large image into tiles …")
    ds = gdal.Open(input_tif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    step = tile_size - overlap
    base = os.path.splitext(os.path.basename(input_tif))[0]
    tiles: List[str] = []
    for y in range(0, ysize, step):
        for x in range(0, xsize, step):
            w = min(tile_size, xsize - x)
            h = min(tile_size, ysize - y)
            out = os.path.join(output_dir, f"{base}_tile_{y}_{x}.tif")
            opts = gdal.TranslateOptions(srcWin=[x, y, w, h], format="GTiff")
            gdal.Translate(out, ds, options=opts).FlushCache()
            tiles.append(out)
    ds = None
    _log(f"{len(tiles)} tile(s) created in {output_dir}")
    return tiles


def mosaic_tiles(tile_paths: List[str], output_path: str) -> None:
    _log("Mosaicking individual tiles …")
    opts = gdal.WarpOptions(format="GTiff")
    gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=tile_paths, options=opts)
    _log(f"Mosaic written to {output_path}")


# -----------------------------------------------------------------------------
# Image preparation / post‑processing
# -----------------------------------------------------------------------------

def prepare_input_rgb(input_path: str, output_dir: str):
    base = os.path.splitext(os.path.basename(input_path))[0]
    _log(f"[{base}] Step 1/3 – preparing input (header + PNG) …")

    ds = gdal.Open(input_path)
    gt = ds.GetGeoTransform()
    xres, yres = gt[1], gt[5]  # pixel sizes

    hdr = os.path.join(output_dir, f"{base}.x4header.tif")
    opts1 = gdal.TranslateOptions(xRes=xres / 4, yRes=yres / 4, resampleAlg="near", format="GTiff")
    gdal.Translate(hdr, ds, options=opts1)
    _log(f"  • Header written: {hdr}")

    png = os.path.join(output_dir, f"{base}.png")
    opts2 = gdal.TranslateOptions(format="PNG")
    gdal.Translate(png, ds, options=opts2)
    _log(f"  • PNG written: {png}")

    ds = None
    return png, hdr


def get_geo_info(src_ds):
    nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    return {
        "nodata": 0 if nodata is None else nodata,
        "size": (src_ds.RasterXSize, src_ds.RasterYSize),
        "gt": src_ds.GetGeoTransform(),
        "proj": src_ds.GetProjection(),
        "dtype": src_ds.GetRasterBand(1).DataType,
    }


def get_min_max(src_ds):
    mins, maxs = [], []
    for i in range(1, src_ds.RasterCount + 1):
        b = src_ds.GetRasterBand(i)
        mn, mx = b.GetMinimum(), b.GetMaximum()
        if mn is None or mx is None:
            b.ComputeStatistics(False)
            mn, mx = b.GetMinimum(), b.GetMaximum()
        mins.append(mn)
        maxs.append(mx)
    return mins, maxs


def write_geotiff_output(png_path: str, header_tif: str, out_tif: str, orig_tif: str):
    base = os.path.splitext(os.path.basename(out_tif))[0]
    _log(f"[{base}] Step 3/3 – writing GeoTIFF …")

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to read super‑resolved PNG")

    # ensure correct channel order (OpenCV: BGR)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hdr_ds = gdal.Open(header_tif)
    info = get_geo_info(hdr_ds)
    xsize, ysize = info["size"]

    if (img.shape[1], img.shape[0]) != (xsize, ysize):
        raise RuntimeError("Size mismatch between PNG and header")

    # build nodata mask from header
    hb = hdr_ds.GetRasterBand(1)
    hdr_nodata = hb.GetNoDataValue() or 0
    mask = hb.ReadAsArray() == hdr_nodata
    hdr_ds = None

    # statistics from original image (unused – preserved for completeness)
    orig_ds = gdal.Open(orig_tif)
    orig_mins, orig_maxs = get_min_max(orig_ds)
    orig_ds = None

    mem = gdal.GetDriverByName("MEM").Create("", xsize, ysize, 3, info["dtype"])
    mem.SetGeoTransform(info["gt"])
    mem.SetProjection(info["proj"])

    for b in range(3):
        arr = img[:, :, b]
        arr[mask] = hdr_nodata
        rb = mem.GetRasterBand(b + 1)
        rb.WriteArray(arr)
        rb.SetNoDataValue(hdr_nodata)

    opts = gdal.TranslateOptions(outputType=info["dtype"], noData=hdr_nodata)
    gdal.Translate(out_tif, mem, options=opts)
    _log(f"GeoTIFF written: {out_tif}")


# -----------------------------------------------------------------------------
# Per‑tile pipeline
# -----------------------------------------------------------------------------

def run_pipeline_on_tile(tile_path: str, output_dir: str, weights: str, tile_idx: int = None, n_tiles: int = None, timings: List[float] = None):
    """Process a single tile. If *tile_idx* and *n_tiles* are provided, progress is logged."""
    name = os.path.splitext(os.path.basename(tile_path))[0]
    prefix = f"[{name}] "
    if tile_idx is not None and n_tiles is not None:
        prefix = f"[Tile {tile_idx}/{n_tiles}] "

    
    # 1. Prepare input (header + PNG)
    png, hdr = prepare_input_rgb(tile_path, output_dir)

    # 2. Inference
    _log(f"{prefix}Step 2/3 – running inference.py …")
    out_png = os.path.join(output_dir, f"{name}.srr.png")

    t0 = time.perf_counter()
    cmd = [sys.executable, "inference.py", "-m", weights, "-i", png, "-o", out_png]
    res = subprocess.run(cmd, capture_output=True, text=True)
    t_infer = time.perf_counter() - t0
    if res.returncode:
        raise RuntimeError(res.stderr)

    if timings is not None:
        timings.append(t_infer)

    _log(f"{prefix}Inference completed in {t_infer:.2f} s  →  {out_png}")

    # 3. Post‑processing – GeoTIFF output
    final = os.path.join(output_dir, f"{name}.srr.tif")
    write_geotiff_output(out_png, hdr, final, tile_path)

    # Return artefacts for optional cleaning
    artefacts = [png, hdr, f"{png}.aux.xml", out_png]
    return final, artefacts


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python sisr_auto_uhr_rgb.py <input> <out_dir> <weights> [-COG] [-CLEAN]")
        sys.exit(1)

    # Manual argument parsing to keep behaviour close to the original implementation
    args = sys.argv[1:]
    cog = "-COG" in args
    clean = "-CLEAN" in args

    for f in ("-COG", "-CLEAN"):
        if f in args:
            args.remove(f)

    if len(args) != 3:
        print("Usage: python sisr_auto_uhr_rgb.py <input> <out_dir> <weights> [-COG] [-CLEAN]")
        sys.exit(1)

    inp, out_dir, weights = args

    _log("Starting UHR RGB super‑resolution pipeline")
    start_time = time.perf_counter()

    # Basic sanity checks ------------------------------------------------------------------
    if not os.path.isfile(inp):
        print("Error: input file not found")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(weights):
        print("Error: weights file not found")
        sys.exit(1)

    ext = os.path.splitext(inp)[1].lower()
    if ext == ".jp2":
        inp = convert_jp2_to_tif(inp, out_dir)

    ds = gdal.Open(inp)
    if ds is None:
        print("Error: cannot open input – aborting")
        sys.exit(1)

    if ds.RasterCount != 3 or gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) != "Byte":
        print("Error: expected 3‑band 8‑bit RGB imagery")
        sys.exit(1)

    wb = os.path.basename(weights)
    if not (wb.startswith("m-5")):
        print("Error: model mismatch for M-5: only weights trained for UHR RGB imagery are supported")
        sys.exit(1)

    X, Y = ds.RasterXSize, ds.RasterYSize
    ds = None

    # Lists to keep track of artefacts for the -CLEAN option ------------------------------
    temp_files: List[str] = []            # all temporary PNG / header / aux files
    per_tile_outputs: List[str] = []      # .srr.tif files for tiles
    generated_tiles: List[str] = []       # raw tile GeoTIFFs
    inference_times: List[float] = []     # individual inference durations

    # Main processing ---------------------------------------------------------------------
    if max(X, Y) > 3000:
        tiles = tile_image(inp, out_dir)
        generated_tiles.extend(tiles)

        n_tiles = len(tiles)
        for idx, tile in enumerate(tiles, 1):
            _log(f"Processing tile {idx}/{n_tiles} …")
            out_tif, artefacts = run_pipeline_on_tile(tile, out_dir, weights, idx, n_tiles, inference_times)
            per_tile_outputs.append(out_tif)
            temp_files.extend(artefacts)

        final = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(inp))[0]}.srr.tif")
        mosaic_tiles(per_tile_outputs, final)
    else:
        _log("Processing single image (no tiling required) …")
        final, artefacts = run_pipeline_on_tile(inp, out_dir, weights, timings=inference_times)
        temp_files.extend(artefacts)

    # Optional conversion to Cloud‑Optimised GeoTIFF --------------------------------------
    if cog:
        _log("Converting final output to Cloud‑Optimised GeoTIFF (COG) …")
        opts = gdal.TranslateOptions(format="COG")
        cog_out = os.path.splitext(final)[0] + ".cog.tif"   # e.g. image.srr.cog.tif
        gdal.Translate(cog_out, final, options=opts)
        _log(f"COG saved: {cog_out}")

    # Cleaning ---------------------------------------------------------------------------
    if clean:
        _log("Removing intermediate artefacts because -CLEAN was specified …")
        _delete_files(temp_files + per_tile_outputs + generated_tiles)
        _log("Temporary files removed")

    # Final timing -----------------------------------------------------------------------
    total_time = time.perf_counter() - start_time
    inference_total = sum(inference_times)

    _log("Processing finished.")
    _log(f"Total processing time : {total_time:.2f} s")
    _log(f"Time spent in inference.py : {inference_total:.2f} s")
    _log(f"Result : {final}")


if __name__ == "__main__":
    main()

