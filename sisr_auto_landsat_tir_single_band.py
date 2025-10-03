import sys, os, subprocess, time, shutil
import numpy as np
import cv2
from osgeo import gdal, gdal_array


def convert_jp2_to_tif(input_path, output_dir):
    print("Step 0: Converting JP2 to GeoTIFF…")
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_tif = os.path.join(output_dir, base + ".tif")
    opts = gdal.TranslateOptions(format="GTiff")
    if gdal.Translate(out_tif, input_path, options=opts) is None:
        raise RuntimeError("Failed to convert JP2 to GeoTIFF.")
    print(f"GeoTIFF saved to: {out_tif}")
    return out_tif


def tile_image(input_tif, output_dir, tile_size=2000, overlap=200):
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
            gdal.Translate(
                out, ds,
                options=gdal.TranslateOptions(srcWin=[x, y, w, h], format="GTiff")
            ).FlushCache()
            tiles.append(out)
    ds = None
    print(f"Created {len(tiles)} tiles.")
    return tiles


def mosaic_tiles(tile_paths, output_path):
    print("Step 4: Merging tiles to final mosaic…")
    gdal.Warp(
        destNameOrDestDS=output_path,
        srcDSOrSrcDSTab=tile_paths,
        options=gdal.WarpOptions(format="GTiff")
    )
    print(f"Final mosaic written to: {output_path}")


def prepare_input_single(input_path, output_dir):
    base = os.path.splitext(os.path.basename(input_path))[0]
    print(f"[{base}] Preparing network input…")
    ds = gdal.Open(input_path)
    gt = ds.GetGeoTransform(); xres, yres = gt[1], gt[5]

    hdr = os.path.join(output_dir, base + ".x4header.tif")
    gdal.Translate(
        hdr, ds,
        options=gdal.TranslateOptions(
            xRes=xres/4, yRes=yres/4,
            resampleAlg="near", format="GTiff"
        )
    )
    print(f"Header written to: {hdr}")

    band = ds.GetRasterBand(1)
    mn, mx = band.GetStatistics(True, True)[:2]
    png = os.path.join(output_dir, base + ".png")
    gdal.Translate(
        png, ds,
        options=gdal.TranslateOptions(
            format="PNG",
            outputType=gdal.GDT_Byte,
            scaleParams=[[mn, mx, 1, 255]]
        )
    )
    print(f"Preview PNG written to: {png}")

    ds = None
    # return paths and their auxiliary files for potential cleanup
    aux_xml = png + ".aux.xml" if os.path.exists(png + ".aux.xml") else None
    tmp_files = [png, hdr, aux_xml] if aux_xml else [png, hdr]
    return png, hdr, tmp_files


def get_geo_info(ds):
    nod = ds.GetRasterBand(1).GetNoDataValue()
    return {
        "nodata": 0 if nod is None else nod,
        "gt":     ds.GetGeoTransform(),
        "proj":   ds.GetProjection(),
        "dtype":  ds.GetRasterBand(1).DataType,
        "size":   (ds.RasterXSize, ds.RasterYSize)
    }


def get_min_max(ds):
    mins, maxs = [], []
    for b in range(1, ds.RasterCount+1):
        band = ds.GetRasterBand(b)
        mn, mx = band.GetMinimum(), band.GetMaximum()
        if mn is None or mx is None:
            band.ComputeStatistics(False)
            mn, mx = band.GetMinimum(), band.GetMaximum()
        mins.append(mn); maxs.append(mx)
    return mins, maxs


def expand_nodata_mask(mask, iterations=1):
    """Dilate the boolean mask by one pixel to erode valid data edges."""
    kernel = np.ones((3, 3), np.uint8)
    mask_uint8 = mask.astype(np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    return dilated.astype(bool)


def write_geotiff_output(png_path, header_tif, out_tif, orig_tif):
    base = os.path.splitext(os.path.basename(out_tif))[0]
    print(f"[{base}] Writing super-resolved GeoTIFF…")

    img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hdr_ds = gdal.Open(header_tif)
    info = get_geo_info(hdr_ds)
    xsize, ysize = info["size"]
    raw_mask = hdr_ds.GetRasterBand(1).ReadAsArray() == info["nodata"]
    hdr_ds = None

    mask = expand_nodata_mask(raw_mask, iterations=1)

    orig_ds = gdal.Open(orig_tif)
    orig_mins, orig_maxs = get_min_max(orig_ds)
    orig_ds = None
    mn, mx = orig_mins[0], orig_maxs[0]

    arr = img.astype(np.float32)
    arr = mn + (arr - 1) * (mx - mn) / 254.0

    arr[mask] = info["nodata"]

    np_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(info["dtype"])
    arr = arr.astype(np_dtype)

    drv = gdal.GetDriverByName("GTiff")
    ds_out = drv.Create(out_tif, xsize, ysize, 1, info["dtype"])
    ds_out.SetGeoTransform(info["gt"])
    ds_out.SetProjection(info["proj"])
    band = ds_out.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(info["nodata"])
    ds_out = None

    print(f"GeoTIFF written to: {out_tif}")


def run_pipeline_on_tile(tile, out_dir, weights):
    base = os.path.splitext(os.path.basename(tile))[0]
    png, hdr, tmp_files = prepare_input_single(tile, out_dir)

    print(f"[{base}] Running neural inference…")
    srr_png = os.path.join(out_dir, base + ".srr.png")
    start = time.time()
    res = subprocess.run(
        ["python", "inference.py", "-m", weights, "-i", png, "-o", srr_png],
        capture_output=True, text=True
    )
    inference_time = time.time() - start
    if res.returncode:
        raise RuntimeError(res.stderr)
    print(f"Inference completed in {inference_time:.1f} s, result: {srr_png}")

    tmp_files.extend([srr_png])

    final = os.path.join(out_dir, base + ".srr.tif")
    write_geotiff_output(srr_png, hdr, final, tile)
    return final, tmp_files, inference_time


def safe_delete(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"Warning: could not delete {p}: {e}")


def main():
    args = sys.argv[1:]
    cog = False
    clean = False
    for flag in ["-COG", "-CLEAN"]:
        if flag in args:
            if flag == "-COG":
                cog = True
            if flag == "-CLEAN":
                clean = True
            args.remove(flag)

    if len(args) != 3:
        print("Usage: python sisr_auto_landsat_tir_single_band.py <input> <out_dir> <weights> [-COG] [-CLEAN]")
        sys.exit(1)
    inp, out_dir, weights = args

    overall_start = time.time()

    print("### Landsat TIR Single-Band Super-Resolution Pipeline ###")
    if not os.path.isfile(inp):
        print("Error: input file not found."); sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(weights):
        print("Error: weights file not found."); sys.exit(1)

    pretrained_base = os.path.splitext(os.path.basename(weights))[0]
    if not pretrained_base.startswith("m-2"):
        print("Model mismatch for M-2: expected weights trained for Landsat TIR imagery.")
        sys.exit(1)

    if inp.lower().endswith(".jp2"):
        inp = convert_jp2_to_tif(inp, out_dir)

    ds = gdal.Open(inp)
    if ds.RasterCount != 1 or gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) not in ("UInt16", "Float32"):
        print("Error: expected single-band 16- or 32-bit input."); sys.exit(1)
    ds = None

    ds2 = gdal.Open(inp)
    X, Y = ds2.RasterXSize, ds2.RasterYSize
    ds2 = None

    inference_total = 0.0
    tmp_to_clean = []
    tile_geotiffs = []

    if max(X, Y) > 3000:
        tiles = tile_image(inp, out_dir)
        total_tiles = len(tiles)
        parts = []
        for idx, t in enumerate(tiles, 1):
            print(f"\n--- Processing tile {idx} of {total_tiles} ---")
            part, tmp_files, inf_t = run_pipeline_on_tile(t, out_dir, weights)
            inference_total += inf_t
            tmp_to_clean.extend(tmp_files)
            parts.append(part)
            tile_geotiffs.append(t)
        final = os.path.join(out_dir, os.path.splitext(os.path.basename(inp))[0] + ".srr.tif")
        mosaic_tiles(parts, final)
        tmp_to_clean.extend(parts)  # per-tile SR outputs
        tmp_to_clean.extend(tile_geotiffs)  # original tiles
    else:
        final, tmp_files, inf_t = run_pipeline_on_tile(inp, out_dir, weights)
        inference_total += inf_t
        tmp_to_clean.extend(tmp_files)

    if cog:
        print("Converting final output to Cloud-Optimized GeoTIFF (COG)…")
        cog_out = os.path.splitext(final)[0] + ".cog.tif"   # create .srr.cog.tif
        gdal.Translate(cog_out, final, options=gdal.TranslateOptions(format="COG"))
        print("COG conversion complete.")

    overall_time = time.time() - overall_start

    if clean:
        print("Cleaning up intermediate files…")
        safe_delete(tmp_to_clean)

    print("\nPipeline completed successfully.")
    print(f"Total inference time: {inference_total:.1f} s")
    print(f"Overall processing time: {overall_time:.1f} s")
    print(f"Final output: {final}")


if __name__ == "__main__":
    main()

