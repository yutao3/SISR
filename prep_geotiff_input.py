import os
import sys
from osgeo import gdal

def is_geotiff(filename):
    """
    Check if the file is a GeoTIFF by ensuring it has a valid geotransform.
    If the projection information is missing, a warning is issued, but processing continues.
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        return False
    geotransform = dataset.GetGeoTransform()
    # Check for the default "identity" geotransform which indicates missing geoinformation.
    valid_geotransform = geotransform != (0, 1, 0, 0, 0, 1)
    if not valid_geotransform:
        return False
    # Warn if the projection information is missing, but do not reject the file.
    if dataset.GetProjection() == '':
        print("Warning: No projection information found, proceeding with geotransform only.")
    return True

def process_image(input_path, output_dir):
    """Process the image as per requirements."""
    print(f"Processing {input_path}...")

    # Check if it's a valid GeoTIFF (now requiring only a valid geotransform)
    if not is_geotiff(input_path):
        print("No valid geotransform found. Exiting.")
        return

    dataset = gdal.Open(input_path)
    band_count = dataset.RasterCount
    nodata = dataset.GetRasterBand(1).GetNoDataValue()

    # Check if the image is of valid type and channels (1 or 3)
    if band_count not in (1, 3):
        print("Error: Unsupported image type. Ensure the image has 1 or 3 channels.")
        return

    # Determine the min and max values for potential scaling
    print("Calculating min and max values...")
    if band_count == 1:
        # Single-band image: take min and max from this band
        stats = dataset.GetRasterBand(1).GetStatistics(True, True)
        min_val, max_val = stats[0], stats[1]
    else:
        # Three-band image: take min and max across all three bands
        min_vals = []
        max_vals = []
        for i in range(1, band_count + 1):
            band_stats = dataset.GetRasterBand(i).GetStatistics(True, True)
            min_vals.append(band_stats[0])
            max_vals.append(band_stats[1])
        min_val = min(min_vals)
        max_val = max(max_vals)

    print(f"Min value: {min_val}, Max value: {max_val}, Nodata: {nodata}")

    # Create the x4header image first, directly from the input, no scaling
    header_filename = os.path.splitext(os.path.basename(input_path))[0] + ".x4header.tif"
    header_path = os.path.join(output_dir, header_filename)
    print(f"Creating x4header file as {header_path}...")

    # Use geotransform to get original x and y resolution.
    x_res_original = dataset.GetGeoTransform()[1]
    y_res_original = dataset.GetGeoTransform()[5]

    gdal.Translate(
        header_path,
        input_path,
        xRes=x_res_original / 4,
        yRes=y_res_original / 4,
        resampleAlg='near'
    )
    print("x4header file created successfully with original min/max/nodata preserved.")

    # Now, create the PNG image.
    # For 1-band: scale from (min_val, max_val) to (1, 255)
    # For 3-band: no scaling, no nodata setting.
    output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".png"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving PNG to {output_path}...")

    if band_count == 1:
        # 1-band: apply scaling to 1-255. Do not set nodata if it doesn't exist.
        scale_params = [[min_val, max_val, 1, 255]]
        translate_options = gdal.TranslateOptions(
            format="PNG",
            outputType=gdal.GDT_Byte,
            scaleParams=scale_params
            # No noData parameter set, respecting original nodata if any
        )
    else:
        # 3-band: no scaling, no nodata changes.
        translate_options = gdal.TranslateOptions(
            format="PNG"
            # No scaleParams, no outputType, no noData
        )

    gdal.Translate(output_path, dataset, options=translate_options)
    print("PNG image saved successfully.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python prep_geotiff_input.py directory/input_image.xxx output_directory")
        sys.exit(1)

    input_image = sys.argv[1]
    output_directory = sys.argv[2]

    # Check if input file exists.
    if not os.path.isfile(input_image):
        print(f"Error: Input file {input_image} does not exist.")
        sys.exit(1)

    # Check and create output directory if it doesn't exist.
    if not os.path.exists(output_directory):
        print(f"Output directory {output_directory} does not exist. Creating directory...")
        os.makedirs(output_directory)
        print("Directory created.")

    # Process the image.
    process_image(input_image, output_directory)

if __name__ == "__main__":
    main()
