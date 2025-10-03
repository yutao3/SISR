# SISR  
**Single-Image Super-Resolution for Earth Observation and Planetary Remote Sensing**

Research code developed by **Dr. Yu Tao (Surrey AI Imaging Limited)** for super-resolving thermal infrared (TIR/MWIR), visible band Earth Observation imagery, and Mars orbital datasets (HiRISE, CTX, HRSC, CaSSIS).  

This repository provides a pipeline to preprocess, upscale, and restore geospatial imagery using the SwinIR architecture. It is intended for **research purposes**.

---

## Features
- Preprocessing and validation of GeoTIFF inputs (EO and planetary datasets)  
- 4× single-image super-resolution using SwinIR  
- Memory-efficient inference for large images  
- Restoration of geospatial metadata in GeoTIFF outputs  
- Example workflows for both Earth and Mars remote sensing imagery  

---

## Installation
Clone this repository and install required dependencies:

```bash
git clone https://github.com/yutao3/SISR.git
cd SISR
pip install -r requirements.txt
```

---

## Quick Start

### Step 1 – Preprocess Input GeoTIFF
Prepare the input image and convert it to PNG format while saving geospatial metadata:

```bash
python prep_geotiff_input.py path/to/input_image.tif path/to/output_dir
```

---

### Step 2 – Run Super-Resolution with SwinIR
Run inference with a pre-trained SwinIR model:

```bash
python inference.py -m path/to/pretrained_model.pth                     -i path/to/output_dir/input_image.png                     -o path/to/output_dir/input_image.srr.png
```

Pre-trained weights are available here:  
🔗 [Google Drive – SISR Models](https://drive.google.com/drive/folders/1KHGWjFf1ZkvSvsjYT-mCSusjUJpetQSW?usp=drive_link)

---

### Step 3 – Restore GeoTIFF Output
Convert the super-resolved PNG back into a GeoTIFF with original metadata:

```bash
python prep_geotiff_output.py path/to/output_dir/input_image.srr.png                               path/to/output_dir/input_image.x4header.tif                               path/to/final_output_image.srr.tif
```

---

## Example Workflow
**Input:** `/data/landsat/LST_206024_20230614.tif`  
**Intermediate directory:** `/data/landsat/tmp`  
**Output:** `/data/landsat/LST_206024_20230614.srr.tif`

```bash
# Step 1 – Preprocess
python prep_geotiff_input.py /data/landsat/LST_206024_20230614.tif /data/landsat/tmp

# Step 2 – Super-resolution inference
python inference.py -m pre-trained-models/m-2_psnr.pth                     -i /data/landsat/tmp/LST_206024_20230614.png                     -o /data/landsat/tmp/LST_206024_20230614.srr.png

# Step 3 – Postprocess and restore metadata
python prep_geotiff_output.py /data/landsat/tmp/LST_206024_20230614.srr.png                               /data/landsat/tmp/LST_206024_20230614.x4header.tif                               /data/landsat/LST_206024_20230614.srr.tif
```

---

## Acknowledgements
- This code builds on the **[SwinIR network architecture](https://github.com/JingyunLiang/SwinIR)** by *Jingyun Liang et al.*  
- Research developed at **Surrey AI Imaging Limited (SAIIL)** for applications in Earth Observation and Mars planetary science.

---

## License
This repository is released for **research and educational purposes**.  
Please contact **Surrey AI Imaging Limited** for any enquiries regarding collaborations or extended usage.

---

## Citation
If you use this code in your research, please cite:

> Tao, Y. (Surrey AI Imaging Limited). *Single-Image Super-Resolution for EO and Mars Imagery (SISR toolbox).* GitHub repository, 2025. https://github.com/yutao3/SISR

---
