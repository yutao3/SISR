import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import network_architecture as net
import warnings  # Import warnings module

# Suppress specific PyTorch warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

def split_image(image, tile_size):
    width, height = image.size
    tiles = []
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            box = (i, j, i + tile_size, j + tile_size)
            tiles.append(image.crop(box))
    return tiles

def stitch_images(tiles, image_size, tile_size):
    print("Starting mosaicing process...")
    full_image = Image.new('RGB', image_size)
    width, height = image_size
    idx = 0
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            full_image.paste(tiles[idx], (i, j))
            idx += 1
    print("Mosaicing completed.")
    return full_image

def define_model(model_path):
    print("Loading model onto GPU for inference...")
    model = net.SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    pretrained_model = torch.load(model_path, weights_only=True)
    model.load_state_dict(pretrained_model.get('params_ema', pretrained_model), strict=True)
    return model

def preprocess_image(path):
    # Read with UNCHANGED to preserve original number of channels
    img_original = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_original is None:
        raise FileNotFoundError(f"Image file not found at path: {path}")

    # Determine if original image is single channel
    if len(img_original.shape) == 2:
        # Single channel, convert to 3-channel for the model
        is_single_channel = True
        img = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    else:
        is_single_channel = False
        img = img_original

    img = img.astype(np.float32) / 255.
    img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    img = torch.from_numpy(img).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
    return img, is_single_channel

def process_small_image(img_lq, model, device):
    print("Processing small image without GPU/CPU memory management...")
    img_lq = img_lq.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_lq)
    return output.cpu()

def process_large_image(img_lq, model_path, device, tile_size=512, tile_overlap=32, batch_size=50):
    b, c, h, w = img_lq.size()
    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h-tile_size, stride)) + [h-tile_size]
    w_idx_list = list(range(0, w-tile_size, stride)) + [w-tile_size]
    sf = 4

    # Initialize E and W on the CPU to save GPU memory
    E = torch.zeros(b, c, h * sf, w * sf).cpu()
    W = torch.zeros_like(E)
    print("Moved accumulation tensors to CPU to reduce GPU memory usage.")

    # Calculate total number of batches and inform the user
    total_tiles = len(h_idx_list) * len(w_idx_list)
    total_batches = (total_tiles + batch_size - 1) // batch_size
    print(f"Total tiles: {total_tiles}. Processing in {total_batches} batches of {batch_size} tiles each.")

    tiles_in_batch = []
    for idx, (h_idx, w_idx) in enumerate([(h, w) for h in h_idx_list for w in w_idx_list]):
        in_patch = img_lq[..., h_idx:h_idx+tile_size, w_idx:w_idx+tile_size].to(device)
        tiles_in_batch.append((in_patch, h_idx, w_idx))

        if (idx + 1) % batch_size == 0 or idx == len(h_idx_list) * len(w_idx_list) - 1:
            print(f"Processing batch {idx // batch_size + 1}/{total_batches} (using GPU)...")

            # Reinitialize model for each batch and move to device
            model = define_model(model_path).to(device)
            model.eval()

            with torch.no_grad():
                for in_patch, h_idx, w_idx in tiles_in_batch:
                    out_patch = model(in_patch).cpu()  # Move output to CPU after inference
                    out_patch_mask = torch.ones_like(out_patch)

                    # Accumulate results on CPU
                    E[..., h_idx*sf:(h_idx+tile_size)*sf, w_idx*sf:(w_idx+tile_size)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile_size)*sf, w_idx*sf:(w_idx+tile_size)*sf].add_(out_patch_mask)

            print("Clearing GPU memory after batch inference to manage limited GPU memory...")
            del model  # Clear model to free memory
            torch.cuda.empty_cache()  # Empty cache to free up GPU memory
            print("GPU memory cleared, switching back to CPU for accumulation.")

            # Clear batch list
            tiles_in_batch.clear()

    output = E.div_(W).to(device)  # Move final output back to GPU if needed for further processing
    print("All tiles processed and merged. Moving final output back to GPU for further processing if required.")
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the pre-trained weights')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input image')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to save the output image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path

    print("Preprocessing image...")
    img_lq, is_single_channel = preprocess_image(args.input)

    # Determine if image is small or large based on tile count
    tile_size = 512
    b, c, h, w = img_lq.size()
    if h <= 2 * tile_size and w <= 2 * tile_size:
        print("Image size detected as small. Using direct processing.")
        model = define_model(model_path)
        output = process_small_image(img_lq, model, device)
    else:
        print("Image size detected as large. Starting inference with GPU memory management for large input image...")
        output = process_large_image(img_lq, model_path, device)

    print("Saving the output image...")
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    # Convert output back to single channel if input was single channel
    if is_single_channel:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(args.output, output)
    print("Output image saved.")

if __name__ == '__main__':
    main()
