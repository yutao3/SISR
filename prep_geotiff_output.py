import sys
import os
import numpy as np
import cv2
from osgeo import gdal
from osgeo import osr

def get_geo_info(src_ds):
    nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
    xsize = src_ds.RasterXSize
    ysize = src_ds.RasterYSize
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    data_type = src_ds.GetRasterBand(1).DataType
    data_type_name = gdal.GetDataTypeName(data_type)
    return nodata_value, xsize, ysize, geotransform, projection, data_type_name, data_type

def get_min_max(src_ds):
    min_values = []
    max_values = []
    band_count = src_ds.RasterCount
    for i in range(1, band_count + 1):
        band = src_ds.GetRasterBand(i)
        min_val = band.GetMinimum()
        max_val = band.GetMaximum()
        # If no precomputed min/max, compute them
        if min_val is None or max_val is None:
            band.ComputeStatistics(0)
            min_val = band.GetMinimum()
            max_val = band.GetMaximum()
        min_values.append(min_val)
        max_values.append(max_val)
    return min_values, max_values

def geo_image(image_filepath, geo_filepath, geo_imagefilepath, original_input_filepath, cog_option=False):
    """
    image_filepath          = path to the upsampled PNG image (SRR result)
    geo_filepath            = path to the upsampled GeoTIFF header file (x4header.tif, 4x bigger than original)
    geo_imagefilepath       = path to final output GeoTIFF (SRR)
    original_input_filepath = path to the original (non-upsampled) GeoTIFF for correct min/max
    cog_option              = whether to produce a COG output
    """
    # Validate paths
    if not os.path.exists(image_filepath):
        sys.exit(1)
    if not os.path.exists(geo_filepath):
        sys.exit(1)
    if not os.path.exists(original_input_filepath):
        sys.exit(1)

    # Load the super-resolved PNG
    input_image = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        sys.exit(1)

    # Convert BGR -> RGB if 3-channel
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Open the 4x upsampled GeoTIFF to obtain georef info (size, transform, projection)
    geo_ds = gdal.Open(geo_filepath)
    if geo_ds is None:
        sys.exit(1)

    nodata_value, xsize, ysize, geotransform, projection, data_type_name, data_type = get_geo_info(geo_ds)
    if nodata_value is None:
        nodata_value = 0

    # Check dimensions match
    if (input_image.shape[1], input_image.shape[0]) != (xsize, ysize):
        # Dimensions mismatch
        sys.exit(1)

    band_count = geo_ds.RasterCount

    # --- NEW LOGIC: read original input for correct min/max ---
    orig_ds = gdal.Open(original_input_filepath)
    if orig_ds is None:
        sys.exit(1)
    orig_min_values, orig_max_values = get_min_max(orig_ds)
    orig_ds_band_count = orig_ds.RasterCount
    # Optionally, you could compare orig_ds_band_count vs. band_count, 
    # but here we just proceed assuming single-band or matching band counts if needed.

    # Convert single gray to BGR if source is 3-band, or vice versa
    if len(input_image.shape) == 2 and band_count == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    elif len(input_image.shape) == 3 and band_count == 1:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Create an in-memory dataset with the same metadata
    mem_driver = gdal.GetDriverByName('MEM')
    if band_count == 1:
        # Single-band
        mem_ds = mem_driver.Create('', xsize, ysize, 1, data_type)
        mem_ds.GetRasterBand(1).WriteArray(input_image)
        mem_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    else:
        # Multi-band
        mem_ds = mem_driver.Create('', xsize, ysize, band_count, data_type)
        for i in range(band_count):
            mem_ds.GetRasterBand(i+1).WriteArray(input_image[:,:,i])
            mem_ds.GetRasterBand(i+1).SetNoDataValue(nodata_value)

    mem_ds.SetGeoTransform(geotransform)
    mem_ds.SetProjection(projection)

    # For single-band images, derive scaleParams from the actual PNG min/max
    # but map it to the min/max of the ORIGINAL GeoTIFF (not the upsampled).
    if band_count == 1:
        png_min = float(input_image.min())
        png_max = float(input_image.max())
        # Use original (non-upsampled) min/max
        scaleParams = [[png_min, png_max, orig_min_values[0], orig_max_values[0]]]
    else:
        scaleParams = None

    # Translate to final output with scaling
    translate_options = gdal.TranslateOptions(
        outputType=data_type,
        noData=nodata_value,
        scaleParams=scaleParams
    )
    gdal.Translate(geo_imagefilepath, mem_ds, options=translate_options)

    # If user requested -COG output, convert to COG
    if cog_option:
        translate_options_cog = gdal.TranslateOptions(format='COG')
        gdal.Translate(geo_imagefilepath, geo_imagefilepath, options=translate_options_cog)

if __name__ == "__main__":
    # Example usage:
    #   python prep_geotiff_output.py <image_filepath> <geo_filepath> <geo_imagefilepath> <original_input_filepath> [-COG]
    args = sys.argv[1:]
    cog_option = False

    # Check for optional -COG argument
    if '-COG' in args:
        cog_option = True
        args.remove('-COG')

    # We now expect 4 mandatory arguments
    if len(args) != 4:
        sys.exit(1)

    image_filepath = args[0]            # path to SRR PNG
    geo_filepath = args[1]             # path to x4header.tif
    geo_imagefilepath = args[2]        # final output path
    original_input_filepath = args[3]  # original, non-upsampled GeoTIFF

    geo_image(image_filepath, geo_filepath, geo_imagefilepath, original_input_filepath, cog_option)

