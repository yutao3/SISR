import sys
import os
import numpy as np
import cv2
from osgeo import gdal
from osgeo import osr

def get_geo_info(src_ds):
    """
    Retrieves georeference info from the source dataset.
    """
    nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    data_type = src_ds.GetRasterBand(1).DataType
    data_type_name = gdal.GetDataTypeName(data_type)
    return nodata_value, xsize, ysize, geotransform, projection, data_type_name, data_type

def get_min_max(src_ds):
    """
    Computes the min and max values across all bands in the dataset.
    """
    min_values = []
    max_values = []
    band_count = src_ds.RasterCount
    for i in range(1, band_count + 1):
        band = src_ds.GetRasterBand(i)
        min_val = band.GetMinimum()
        max_val = band.GetMaximum()
        if min_val is None or max_val is None:
            print(f"Computing statistics for band {i}.")
            band.ComputeStatistics(0)
            min_val = band.GetMinimum()
            max_val = band.GetMaximum()
        min_values.append(min_val)
        max_values.append(max_val)
    overall_min = min(min_values)
    overall_max = max(max_values)
    return overall_min, overall_max

def geo_image(image_filepath, geo_filepath, geo_imagefilepath, cog_option=False):
    print(f"Opening input image: {image_filepath}")
    if not os.path.exists(image_filepath):
        print(f"Error: Input image {image_filepath} does not exist.")
        sys.exit(1)
    if not os.path.exists(geo_filepath):
        print(f"Error: x4header file {geo_filepath} does not exist.")
        sys.exit(1)

    # Open input image
    input_image = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        print("Error: Could not read input image.")
        sys.exit(1)
    print(f"Input image shape: {input_image.shape}")

    # Open x4header image
    geo_ds = gdal.Open(geo_filepath)
    if geo_ds is None:
        print("Error: Could not open x4header file.")
        sys.exit(1)

    nodata_value, xsize, ysize, geotransform, projection, data_type_name, data_type = get_geo_info(geo_ds)
    print(f"NoData value from x4header: {nodata_value}")
    print(f"Data type from x4header: {data_type_name}")

    if nodata_value is None:
        nodata_value = 0
        print("Warning: NoData value is None in x4header, setting to 0.")

    if (input_image.shape[1], input_image.shape[0]) != (xsize, ysize):
        print("Error: Input image and x4header image have different dimensions.")
        sys.exit(1)

    # Adjust NoData pixels in input image (pixels with value 0)
    print("Adjusting NoData pixels in input image.")
    input_image[input_image == 0] = nodata_value

    # Get min and max values from x4header
    print("Computing min and max values from x4header image.")
    min_x4header, max_x4header = get_min_max(geo_ds)
    print(f"Min value in x4header image: {min_x4header}")
    print(f"Max value in x4header image: {max_x4header}")

    if min_x4header == max_x4header:
        print("Error: Min and max values of x4header image are equal.")
        sys.exit(1)

    # Check if x4header is single-channel
    band_count = geo_ds.RasterCount
    print(f"Number of bands in x4header image: {band_count}")

    # Convert input image to match x4header band count
    if band_count == 1 and len(input_image.shape) == 3:
        print("Converting input image to single-channel using OpenCV.")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    elif band_count == 3 and len(input_image.shape) == 2:
        print("Error: Input image is single-channel but x4header is three-channel.")
        sys.exit(1)

    # Create a mask for NoData pixels
    print("Creating mask for NoData pixels.")
    nodata_mask = (input_image == nodata_value)

    # Rescale input image from (1,255) to (min_x4header, max_x4header)
    print("Rescaling input image to match x4header min and max values.")
    scale_factor = (max_x4header - min_x4header) / (255 - 1)
    input_image = input_image.astype(np.float32)
    input_image_rescaled = np.full_like(input_image, nodata_value, dtype=np.float32)

    # Apply rescaling only to valid pixels
    valid_mask = ~nodata_mask
    input_image_rescaled[valid_mask] = (input_image[valid_mask] - 1) * scale_factor + min_x4header

    # Clip values to valid range
    print("Clipping rescaled image to valid data type range.")
    data_type_ranges = {
        'Byte': (0, 255),
        'UInt16': (0, 65535),
        'Int16': (-32768, 32767),
        'UInt32': (0, 4294967295),
        'Int32': (-2147483648, 2147483647),
        'Float32': (-3.4e38, 3.4e38),
        'Float64': (-1.7e308, 1.7e308)
    }
    min_valid_value, max_valid_value = data_type_ranges.get(data_type_name, (-np.inf, np.inf))
    input_image_rescaled = np.clip(input_image_rescaled, min_valid_value, max_valid_value)

    # Convert data type
    print("Converting rescaled image to match x4header data type.")
    data_type_mapping = {
        'Byte': np.uint8,
        'UInt16': np.uint16,
        'Int16': np.int16,
        'UInt32': np.uint32,
        'Int32': np.int32,
        'Float32': np.float32,
        'Float64': np.float64
    }
    np_data_type = data_type_mapping.get(data_type_name, np.float32)
    input_image_rescaled = input_image_rescaled.astype(np_data_type)

    # Create output geotiff
    print("Creating output geotiff image.")
    driver = gdal.GetDriverByName('GTiff')
    if driver is None:
        print("Error: GTiff driver not available.")
        sys.exit(1)

    num_bands = 1 if len(input_image_rescaled.shape) == 2 else input_image_rescaled.shape[2]

    # If COG option is specified, create a temporary output file
    if cog_option:
        temp_output_file = 'temp_output.tif'
        output_file = temp_output_file
    else:
        output_file = geo_imagefilepath

    output_ds = driver.Create(output_file, xsize, ysize, num_bands, data_type)
    if output_ds is None:
        print("Error: Could not create output dataset.")
        sys.exit(1)

    # Set geotransform and projection
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)

    # Write data to output dataset
    print("Writing data to output geotiff.")
    if num_bands == 1:
        output_ds.GetRasterBand(1).WriteArray(input_image_rescaled)
        output_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    else:
        for band in range(num_bands):
            output_ds.GetRasterBand(band + 1).WriteArray(input_image_rescaled[:, :, band])
            output_ds.GetRasterBand(band + 1).SetNoDataValue(nodata_value)

    output_ds.FlushCache()
    output_ds = None
    print(f"Output geotiff image created: {output_file}")

    if cog_option:
        print("Converting output to Cloud Optimized GeoTIFF (COG).")
        translate_options = gdal.TranslateOptions(format='COG')
        gdal.Translate(geo_imagefilepath, temp_output_file, options=translate_options)
        print(f"COG image created: {geo_imagefilepath}")
        # Remove temporary file
        os.remove(temp_output_file)

if __name__ == "__main__":
    args = sys.argv[1:]
    cog_option = False

    if '-COG' in args:
        cog_option = True
        args.remove('-COG')

    if len(args) != 3:
        print("Usage: python prep_geotiff_output.py input_image.xxx x4header.tif output_image.tif [-COG]")
        sys.exit(1)

    image_filepath = args[0]
    geo_filepath = args[1]
    geo_imagefilepath = args[2]

    geo_image(image_filepath, geo_filepath, geo_imagefilepath, cog_option)

