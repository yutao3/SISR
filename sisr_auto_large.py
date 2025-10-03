import sys
import os
import subprocess
from osgeo import gdal

def tile_image(input_tif, output_dir, tile_size=2000, overlap=200):
    """
    Splits the input GeoTIFF into smaller tiles (tile_size x tile_size) with
    the specified overlap in both X and Y dimensions.

    Returns a list of the tile file paths.
    """
    ds = gdal.Open(input_tif)
    if ds is None:
        raise RuntimeError(f"Could not open {input_tif} for tiling.")

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    tile_paths = []

    # We step by (tile_size - overlap).
    x_step = tile_size - overlap
    y_step = tile_size - overlap

    base_name = os.path.splitext(os.path.basename(input_tif))[0]

    for y_off in range(0, ysize, y_step):
        for x_off in range(0, xsize, x_step):
            # Compute the actual size for this tile (to not exceed the image boundary).
            w = min(tile_size, xsize - x_off)
            h = min(tile_size, ysize - y_off)

            # GDAL's srcWin = [xoff, yoff, xsize, ysize]
            # This tile's filename
            tile_name = f"{base_name}_tile_{y_off}_{x_off}.tif"
            tile_path = os.path.join(output_dir, tile_name)

            # Create the tile
            translate_opts = gdal.TranslateOptions(
                srcWin=[x_off, y_off, w, h],
                format="GTiff"
            )
            # Capture the returned dataset and then close it.
            tile_ds = gdal.Translate(tile_path, ds, options=translate_opts)
            tile_ds = None  # Ensure the tile is flushed to disk

            tile_paths.append(tile_path)

    ds = None
    return tile_paths

def mosaic_tiles(tile_paths, output_path):
    """
    Mosaics the list of GeoTIFF tiles into a single output GeoTIFF.
    """
    # Use gdal.Warp for mosaicking; it handles overlaps gracefully
    vrt_options = gdal.WarpOptions(format='GTiff')
    gdal.Warp(destNameOrDestDS=output_path, srcDSOrSrcDSTab=tile_paths, options=vrt_options)


def run_pipeline_on_tile(tile_path, output_dir, pretrained_weights):
    """
    Runs the three-step pipeline (prep_geotiff_input.py, inference.py, prep_geotiff_output.py)
    on a single tile, returning the path to the final super-resolved GeoTIFF of that tile.
    """
    base_name = os.path.splitext(os.path.basename(tile_path))[0]

    # Step 1: prep_geotiff_input.py
    print(f"[Tile {base_name}] Step 1: Running prep_geotiff_input.py...")
    cmd1 = ["python", "prep_geotiff_input.py", tile_path, output_dir]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        raise RuntimeError(f"Error during prep_geotiff_input.py on {tile_path}:\n{result1.stderr}")

    # Check expected outputs
    input_png = os.path.join(output_dir, base_name + ".png")
    x4header_tif = os.path.join(output_dir, base_name + ".x4header.tif")
    if not os.path.isfile(input_png) or not os.path.isfile(x4header_tif):
        raise RuntimeError(f"Missing .png or .x4header.tif for {tile_path}")

    # Step 2: inference.py
    print(f"[Tile {base_name}] Step 2: Running inference.py...")
    srr_png = os.path.join(output_dir, base_name + ".srr.png")
    cmd2 = ["python", "inference.py", "-m", pretrained_weights, "-i", input_png, "-o", srr_png]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode != 0:
        raise RuntimeError(f"Error during inference.py on {tile_path}:\n{result2.stderr}")

    if not os.path.isfile(srr_png):
        raise RuntimeError(f"Missing .srr.png for {tile_path}")

    # Step 3: prep_geotiff_output.py
    print(f"[Tile {base_name}] Step 3: Running prep_geotiff_output.py...")
    srr_tif = os.path.join(output_dir, base_name + ".srr.tif")
    cmd3 = ["python", "prep_geotiff_output.py", srr_png, x4header_tif, srr_tif, tile_path]
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    if result3.returncode != 0:
        raise RuntimeError(f"Error during prep_geotiff_output.py on {tile_path}:\n{result3.stderr}")

    if not os.path.isfile(srr_tif):
        raise RuntimeError(f"Missing final .srr.tif for {tile_path}")

    print(f"[Tile {base_name}] Processing completed successfully.")
    return srr_tif


def main():
    if len(sys.argv) != 4:
        print("Usage: python sisr_auto.py <input_image> <output_directory> <pretrained_weights>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_dir = sys.argv[2]
    pretrained_weights = sys.argv[3]

    # Validate inputs
    if not os.path.isfile(input_image):
        print(f"Error: Input file {input_image} does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating directory...")
        os.makedirs(output_dir)

    if not os.path.isfile(pretrained_weights):
        print(f"Error: Pre-trained weights file {pretrained_weights} does not exist.")
        sys.exit(1)

    # If input is JP2, convert to GeoTIFF
    input_ext = os.path.splitext(input_image)[1].lower()
    if input_ext == ".jp2":
        print("Input image is JP2 format. Converting to GeoTIFF...")
        jp2_base = os.path.splitext(os.path.basename(input_image))[0]
        converted_geotiff = os.path.join(output_dir, jp2_base + ".tif")

        translate_options = gdal.TranslateOptions(format="GTiff")
        result_ds = gdal.Translate(converted_geotiff, input_image, options=translate_options)
        if result_ds is None:
            print("Error: Failed to convert JP2 to GeoTIFF.")
            sys.exit(1)
        input_geotiff = converted_geotiff
        print(f"Conversion successful. Using {input_geotiff} as input.")
    else:
        input_geotiff = input_image

    # Check the input GeoTIFF
    ds = gdal.Open(input_geotiff)
    if ds is None:
        print("Error: Could not open input GeoTIFF.")
        sys.exit(1)
    band_count = ds.RasterCount
    data_type = ds.GetRasterBand(1).DataType
    data_type_name = gdal.GetDataTypeName(data_type)

    # Validate model and image conditions
    pretrained_base = os.path.basename(pretrained_weights)
    if band_count == 3:
        # Must be 8-bit
        if data_type_name != 'Byte':
            print("Input image format is not supported. Please ensure 3-band RGB is Byte.")
            sys.exit(1)
        # Optional checks: m-1 or m-5
        if not (pretrained_base.startswith("m-1") or pretrained_base.startswith("m-5")):
            print("Model mismatch for 3-band images.")
            sys.exit(1)

    elif band_count == 1:
        # Must be Float32 or UInt16
        if data_type_name not in ['Float32', 'UInt16']:
            print("Input image format is not supported. Please ensure single-band geotiff is Float32 or UInt16.")
            sys.exit(1)
        # Optional checks: m-2
        if not pretrained_base.startswith("m-2"):
            print("Model mismatch for single-band images.")
            sys.exit(1)
    else:
        print("Input geotif image format is not supported. Must be single-band Float32/UInt16 or three-band Byte.")
        sys.exit(1)

    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    base_name = os.path.splitext(os.path.basename(input_geotiff))[0]
    ds = None

    # Decide whether to tile or run single pass
    if x_size > 3000 or y_size > 3000:
        print("Large image detected. Splitting into tiles...")

        # 1. Tile
        tile_paths = tile_image(input_geotiff, output_dir, tile_size=2000, overlap=200)

        # 2. Process each tile
        srr_tiles = []
        for tile_path in tile_paths:
            srr_tile = run_pipeline_on_tile(tile_path, output_dir, pretrained_weights)
            srr_tiles.append(srr_tile)

        # 3. Mosaic the super-resolved tiles
        final_srr_tif = os.path.join(output_dir, base_name + ".srr.tif")
        print("Mosaicking tiles...")
        mosaic_tiles(srr_tiles, final_srr_tif)

        if not os.path.isfile(final_srr_tif):
            print("Error: Final mosaic failed.")
            sys.exit(1)

        print(f"Processing completed successfully. The super-resolution GeoTIFF is located at {final_srr_tif}.")

    else:
        # Process as before (no tiling)
        print("Image size is manageable. Processing in one pass.")

        # Step 1: prep_geotiff_input.py
        print("Step 1: Running prep_geotiff_input.py...")
        cmd1 = ["python", "prep_geotiff_input.py", input_geotiff, output_dir]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        if result1.returncode != 0:
            print("Error during prep_geotiff_input.py execution:")
            print(result1.stderr)
            sys.exit(1)

        # Check if expected outputs exist
        input_png = os.path.join(output_dir, base_name + ".png")
        x4header_tif = os.path.join(output_dir, base_name + ".x4header.tif")
        if not os.path.isfile(input_png) or not os.path.isfile(x4header_tif):
            print("Error: Expected output files from prep_geotiff_input.py are missing.")
            sys.exit(1)
        print("Step 1 completed successfully.")

        # Step 2: inference.py
        print("Step 2: Running inference.py...")
        srr_png = os.path.join(output_dir, base_name + ".srr.png")
        cmd2 = ["python", "inference.py", "-m", pretrained_weights, "-i", input_png, "-o", srr_png]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print("Error during inference.py execution:")
            print(result2.stderr)
            sys.exit(1)

        if not os.path.isfile(srr_png):
            print("Error: SRR PNG file from inference.py is missing.")
            sys.exit(1)
        print("Step 2 completed successfully.")

        # Step 3: prep_geotiff_output.py
        print("Step 3: Running prep_geotiff_output.py...")
        srr_tif = os.path.join(output_dir, base_name + ".srr.tif")
        cmd3 = ["python", "prep_geotiff_output.py", srr_png, x4header_tif, srr_tif, input_geotiff]
        result3 = subprocess.run(cmd3, capture_output=True, text=True)
        if result3.returncode != 0:
            print("Error during prep_geotiff_output.py execution:")
            print(result3.stderr)
            sys.exit(1)

        if not os.path.isfile(srr_tif):
            print("Error: Final SRR TIFF file from prep_geotiff_output.py is missing.")
            sys.exit(1)
        print("Step 3 completed successfully.")

        # All steps succeeded
        print(f"Processing completed successfully. The super-resolution GeoTIFF is located at {srr_tif}.")


if __name__ == "__main__":
    main()

