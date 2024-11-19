# SISR
A Pre-trained Single-Image Super-Resolution Toolbox for Mars and Earth Observations

## Quick Start Guide for Single-Image Super-Resolution (SISR) Pipeline

To enhance the resolution of your geospatial images using the SISR pipeline with the SwinIR model, follow these three key steps:

1. Preprocess the Input Image:
   Use the `prep_geotiff_input.py` script to prepare your GeoTIFF image for super-resolution. This script validates that the input is a proper GeoTIFF with geospatial metadata, rescales pixel values to an appropriate range, and converts the image to PNG format suitable for the model. Run the following command:
   
python prep_geotiff_input.py path/to/your_input_image.tif path/to/output_directory

2. Run Inference with the SwinIR Model:
Execute the `inference.py` script to perform super-resolution (4x upscaling) on the preprocessed image using your pre-trained SwinIR model. This script efficiently handles both small and large images by managing memory during processing. Run the following command:

python inference.py -m path/to/pre-trained_model.pth -i path/to/output_directory/your_input_image.png -o path/to/output_directory/your_output_image.srr.png

3. Post-process and Save the Output Image:
Use the `prep_geotiff_output.py` script to convert the super-resolved PNG image back to a GeoTIFF, restoring the original geospatial metadata. Run the following command:

python prep_geotiff_output.py path/to/output_directory/your_output_image.srr.png path/to/output_directory/your_input_image.x4header.tif path/to/your_final_output_image.srr.tif

## Example Usage of the SISR Pipeline

Suppose you have an input GeoTIFF image named `LST_206024_20230614.tif` located in the `/data/landsat/` directory. You want to process this image using the SISR pipeline and store intermediate files in a temporary directory at `/data/landsat/tmp`. The final super-resolved image will be saved as `/data/landsat/LST_206024_20230614.srr.tif`. Here's how you can achieve this:

1. Preprocess the Input Image:
Use the `prep_geotiff_input.py` script to prepare your GeoTIFF image for super-resolution. This script checks for geospatial metadata, rescales pixel values, and converts the image to PNG format suitable for the model.

python prep_geotiff_input.py /data/landsat/LST_206024_20230614.tif /data/landsat/tmp

**Explanation:** This command takes the input image `/data/landsat/LST_206024_20230614.tif` and processes it, saving the rescaled PNG image and a header file in the temporary directory `/data/landsat/tmp`.

2. Run Inference with the SwinIR Model:
Execute the `inference.py` script to perform super-resolution on the preprocessed image. Ensure you specify the path to your pre-trained SwinIR model (`m-2_psnr.pth` in this example).

python inference.py -m pre-trained-models/m-2_psnr.pth -i /data/landsat/tmp/LST_206024_20230614.png -o /data/landsat/tmp/LST_206024_20230614.srr.png

**Explanation:** This command loads the pre-trained model from `pre-trained-models/m-2_psnr.pth` and processes the preprocessed image `/data/landsat/tmp/LST_206024_20230614.png`, saving the super-resolved PNG image as `/data/landsat/tmp/LST_206024_20230614.srr.png`.

3. Post-process and Save the Output Image:
Use the `prep_geotiff_output.py` script to convert the super-resolved PNG image back to a GeoTIFF, restoring the original geospatial metadata.

python prep_geotiff_output.py /data/landsat/tmp/LST_206024_20230614.srr.png /data/landsat/tmp/LST_206024_20230614.x4header.tif /data/landsat/LST_206024_20230614.srr.tif

**Explanation:** This command takes the super-resolved PNG image and the header file (containing geospatial information) from the temporary directory and generates the final GeoTIFF image `/data/landsat/LST_206024_20230614.srr.tif`.
