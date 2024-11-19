import os
import sys
from osgeo import gdal

def is_geotiff(filename):
    """Check if the file is a GeoTIFF by looking for spatial reference and geotransform information."""
    dataset = gdal.Open(filename)
    if dataset is None:
        return False
    has_projection = dataset.GetProjection() != ''
    has_geotransform = dataset.GetGeoTransform() != (0, 1, 0, 0, 0, 1)  # Default "identity" geotransform
    return has_projection and has_geotransform

def process_image(input_path, output_dir):
    """Process the image by validating type, rescaling, and saving as per requirements."""
    print(f"Processing {input_path}...")

    # Check if it's a valid GeoTIFF
    if not is_geotiff(input_path):
        print("No geoinformation found. Exiting.")
        return

    dataset = gdal.Open(input_path)
    band_count = dataset.RasterCount
    nodata = dataset.GetRasterBand(1).GetNoDataValue()

    # Check if the image is of valid type and channels
    if band_count not in (1, 3):
        print("Error: Unsupported image type. Ensure the image has 1 or 3 channels.")
        return

    # Determine the min and max values for scaling
    print("Calculating min and max values for rescaling...")
    stats = dataset.GetRasterBand(1).GetStatistics(True, True)
    min_val, max_val = stats[0], stats[1]
    if nodata is not None:
        min_val = min(min_val, 1)  # Avoid nodata affecting min
    print(f"Min value: {min_val}, Max value: {max_val}")

    # Convert filename to PNG format and save using gdal_translate
    output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".png"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving rescaled image as PNG to {output_path}...")

    translate_options = gdal.TranslateOptions(
        format="PNG",
        outputType=gdal.GDT_Byte,
        scaleParams=[[min_val, max_val, 1, 255]],
        noData=0
    )
    gdal.Translate(output_path, dataset, options=translate_options)
    print("Image saved successfully.")

    # Create x4 header image
    header_filename = os.path.splitext(os.path.basename(input_path))[0] + ".x4header.tif"
    header_path = os.path.join(output_dir, header_filename)
    print(f"Creating x4 header file as {header_path}...")
    gdal.Translate(header_path, input_path, xRes=dataset.GetGeoTransform()[1] / 4, yRes=-dataset.GetGeoTransform()[5] / 4)
    print("x4 header file created successfully.")

def main():
    # Basic input validation
    if len(sys.argv) != 3:
        print("Usage: python prep_geotiff_input.py directory/input_image.xxx output_directory")
        sys.exit(1)

    input_image = sys.argv[1]
    output_directory = sys.argv[2]

    # Check if input file exists
    if not os.path.isfile(input_image):
        print(f"Error: Input file {input_image} does not exist.")
        sys.exit(1)

    # Check and create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        print(f"Output directory {output_directory} does not exist. Creating directory...")
        os.makedirs(output_directory)
        print("Directory created.")

    # Process the image
    process_image(input_image, output_directory)

if __name__ == "__main__":
    main()

