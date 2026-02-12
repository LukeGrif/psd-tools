import os
from psd_tools import PSDImage
import numpy as np
import cv2

# ---------- CONFIG ----------
PSD_PATH = "0126_2_ADS_0939.psd"   # input PSD file
OUTPUT_DIR = "psd_tiled_output_to_test"    # where tiles will be saved

TILE_SIZE = 640
STRIDE = 320

# ----------------------------

def psd_layer1_to_image(psd_path):
    """
    Load ONLY the first visible layer (Layer 1) from the PSD
    and convert it to an OpenCV BGR image.
    """
    psd = PSDImage.open(psd_path)

    # Get first layer (index 0)
    layer = psd[1]

    # Render just this layer
    pil_img = layer.composite()
    img = np.array(pil_img)

    # Convert RGB/RGBA → BGR for OpenCV
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def tile_image(image, base_name):
    """Split image into overlapping tiles and save them."""

    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    h, w = image.shape[:2]
    tile_id = 0

    for y in range(0, h - TILE_SIZE + 1, STRIDE):
        for x in range(0, w - TILE_SIZE + 1, STRIDE):

            tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]

            out_img_name = f"{base_name}_tile{tile_id}.jpg"
            out_path = os.path.join(out_img_dir, out_img_name)

            cv2.imwrite(out_path, tile)
            tile_id += 1

    print(f"Saved {tile_id} tiles → {out_img_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Convert PSD Layer 1 → image ---
    image = psd_layer1_to_image(PSD_PATH)

    # --- Save full JPG for reference ---
    full_jpg_path = os.path.join(OUTPUT_DIR, "full_image.jpg")
    cv2.imwrite(full_jpg_path, image)
    print("Saved Layer‑1 JPG →", full_jpg_path)

    # --- Tile image ---
    base_name = os.path.splitext(os.path.basename(PSD_PATH))[0]
    tile_image(image, base_name)


if __name__ == "__main__":
    main()
