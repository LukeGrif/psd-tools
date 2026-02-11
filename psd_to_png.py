from psd_tools import PSDImage
import cv2
import numpy as np
import os


INPUT = "image_to_test/0126_1_ADS_0934.psd"


def main():
    psd = PSDImage.open(INPUT)

    print("\nAvailable visible layers:\n")

    visible_layers = []
    for i, layer in enumerate(psd):
        if layer.is_visible():
            visible_layers.append(layer)
            print(f"[{len(visible_layers)-1}] {layer.name}")

    if not visible_layers:
        print("No visible layers found.")
        return

    # ---------- ask user ----------
    while True:
        choice = input("\nEnter layer number to export as PNG: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(visible_layers):
            break
        print("Invalid number, try again.")

    selected_layer = visible_layers[int(choice)]

    print(f"\nUsing layer: {selected_layer.name}")

    # ---------- render layer ----------
    img = selected_layer.composite()
    if img is None:
        print("Could not render this layer.")
        return

    # ---------- convert to OpenCV ----------
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    # ---------- output path ----------
    output_path = os.path.splitext(INPUT)[0] + f"_{selected_layer.name}.png"

    cv2.imwrite(output_path, bgr)

    print(f"Saved PNG → {output_path}")


if __name__ == "__main__":
    main()
