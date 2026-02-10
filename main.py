import os
import cv2
import numpy as np
from psd_tools import PSDImage
import re


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def is_numeric(name: str) -> bool:
    """Return True if layer name is purely a number."""
    return name.strip().isdigit()


def natural_key(text):
    """
    Sort helper that handles numbers correctly.
    Example: image2 < image10
    """
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r'(\d+)', text)
    ]


def choose_background_layer(psd):
    """
    Ask user to choose which layer is the real background image.
    Only non-numeric visible layers are shown.
    Keeps prompting until valid input is given.
    """
    candidates = []

    print("\nSelect the BACKGROUND layer:\n")

    for i, layer in enumerate(psd):
        if not layer.is_visible():
            continue
        if is_numeric(layer.name):
            continue

        candidates.append((i, layer))
        print(f"[{len(candidates)-1}] PSD index {i} → '{layer.name}'")

    if not candidates:
        raise RuntimeError("No valid background layer candidates found.")

    # --- SAFE INPUT LOOP ---
    while True:
        user_input = input("\nEnter number of correct background layer: ").strip()

        # check numeric
        if not user_input.isdigit():
            print("Please enter a valid NUMBER from the list.")
            continue

        choice = int(user_input)

        # check range
        if choice < 0 or choice >= len(candidates):
            print("Number out of range. Try again.")
            continue

        # valid choice
        _, chosen_layer = candidates[choice]
        break

    img = chosen_layer.composite()
    if img is None:
        raise RuntimeError("Chosen background layer could not be rendered.")

    print(f"Using background layer: '{chosen_layer.name}'\n")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)



def make_padded_square_bbox(x, y, w, h, img_w, img_h, pad_ratio=0.15):
    """Expand bbox by percentage and convert to square."""
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x -= pad_w
    y -= pad_h
    w += 2 * pad_w
    h += 2 * pad_h

    side = max(w, h)

    cx = x + w // 2
    cy = y + h // 2

    x = cx - side // 2
    y = cy - side // 2
    w = h = side

    x = max(0, x)
    y = max(0, y)

    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y

    return x, y, w, h


# ------------------------------------------------------------
# Core processing for ONE PSD
# ------------------------------------------------------------

def process_psd(psd_path, output_root, pad_ratio=0.15):
    psd = PSDImage.open(psd_path)

    psd_name = os.path.splitext(os.path.basename(psd_path))[0]
    output_dir = os.path.join(output_root, psd_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Processing: {psd_name} ===")
    print(f"Size: {psd.width} x {psd.height}")
    print(f"Layers: {len(psd)}")

    # --- user selects background ---
    background = choose_background_layer(psd)
    img_h, img_w = background.shape[:2]

    # --- iterate annotation layers ---
    for layer in psd:

        if not layer.is_visible():
            continue

        if is_numeric(layer.name):
            continue

        pil_img = layer.composite()
        if pil_img is None:
            continue

        layer_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        alpha = layer_img[:, :, 3]

        if np.max(alpha) == 0:
            continue

        # --- contours in LOCAL coords ---
        _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"Layer: {layer.name} → {len(contours)} objects")

        # --- PSD global offset ---
        x0, y0, _, _ = layer.bbox

        for j, cnt in enumerate(contours):

            cnt_global = cnt + np.array([[x0, y0]])

            x, y, w, h = cv2.boundingRect(cnt_global)

            x, y, w, h = make_padded_square_bbox(
                x, y, w, h, img_w, img_h, pad_ratio
            )

            cropped_img = background[y:y+h, x:x+w].copy()

            safe_name = layer.name.replace("/", "_")
            out_path = os.path.join(
                output_dir, f"{safe_name}_obj{j+1}.png"
            )

            cv2.imwrite(out_path, cropped_img)

    print(f"Saved outputs → {output_dir}")


# ------------------------------------------------------------
# Batch folder processing
# ------------------------------------------------------------

def process_folder(input_folder, output_root="output", pad_ratio=0.15):
    psd_files = sorted(
    [f for f in os.listdir(input_folder) if f.lower().endswith(".psd")],
    key=natural_key
    )

    if not psd_files:
        print("No PSD files found.")
        return

    print(f"\nFound {len(psd_files)} PSD files.\n")

    for psd_file in psd_files:
        psd_path = os.path.join(input_folder, psd_file)
        process_psd(psd_path, output_root, pad_ratio)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":

    INPUT_FOLDER = "Images"   # folder containing PSD files
    OUTPUT_FOLDER = "output"
    PAD_RATIO = 0.15          # 10–20% typical

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, PAD_RATIO)
