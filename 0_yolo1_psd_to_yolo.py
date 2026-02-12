import os
import cv2
import numpy as np
import re
from psd_tools import PSDImage


class PSDToYOLO:
    def __init__(self, input_folder, output_root="output_yolo", pad_ratio=0.15):
        self.input_folder = input_folder
        self.output_root = output_root
        self.pad_ratio = pad_ratio
        self.class_map = {}

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    @staticmethod
    def is_numeric(name: str) -> bool:
        return name.strip().isdigit()

    @staticmethod
    def natural_key(text):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    @staticmethod
    def make_padded_square_bbox(x, y, w, h, img_w, img_h, pad_ratio):
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)

        x -= pad_w
        y -= pad_h
        w += 2 * pad_w
        h += 2 * pad_h

        side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2

        x = max(0, cx - side // 2)
        y = max(0, cy - side // 2)

        w = min(side, img_w - x)
        h = min(side, img_h - y)

        return x, y, w, h

    @staticmethod
    def bbox_to_yolo(x, y, w, h, img_w, img_h):
        return (
            (x + w / 2) / img_w,
            (y + h / 2) / img_h,
            w / img_w,
            h / img_h,
        )

    # ------------------------------------------------------------
    # PASS 1 → Collect user choices
    # ------------------------------------------------------------

    def collect_layer_choices(self, psd_files):
        choices = {}

        print("\n========== SETUP PHASE ==========\n")

        for psd_file in psd_files:

            psd_path = os.path.join(self.input_folder, psd_file)
            psd = PSDImage.open(psd_path)
            name = os.path.splitext(psd_file)[0]

            print(f"\n--- {name} ---")

            candidates = [
                layer for layer in psd
                if layer.is_visible() and not self.is_numeric(layer.name)
            ]

            for i, layer in enumerate(candidates):
                print(f"[{i}] {layer.name}")

            # background
            while True:
                bg = input("Background layer #: ").strip()
                if bg.isdigit() and 0 <= int(bg) < len(candidates):
                    break
                print("Invalid.")

            bg_layer = candidates[int(bg)]

            # ignored
            ignore_input = input("Ignore layer numbers (comma or blank): ").strip()
            ignored = []

            if ignore_input:
                for val in ignore_input.split(","):
                    val = val.strip()
                    if val.isdigit() and 0 <= int(val) < len(candidates):
                        ignored.append(candidates[int(val)].name)

            choices[name] = (bg_layer.name, ignored)

        print("\n========== SETUP COMPLETE ==========\n")
        return choices

    # ------------------------------------------------------------
    # PASS 2 → Process PSD
    # ------------------------------------------------------------

    def process_single_psd(self, psd_path, layer_choices):

        psd = PSDImage.open(psd_path)
        name = os.path.splitext(os.path.basename(psd_path))[0]

        bg_name, ignored_names = layer_choices[name]

        print(f"\n=== Processing {name} ===")

        # find background layer
        bg_layer = next((l for l in psd if l.name == bg_name), None)
        if bg_layer is None:
            raise RuntimeError(f"Background layer '{bg_name}' not found in {name}")

        # render background
        bg_img = bg_layer.composite()
        background = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGBA2BGR)

        img_h, img_w = background.shape[:2]

        # output dirs
        images_dir = os.path.join(self.output_root, "images")
        labels_dir = os.path.join(self.output_root, "labels")
        preview_dir = os.path.join(self.output_root, "preview")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)

        # save base image
        cv2.imwrite(os.path.join(images_dir, f"{name}.png"), background)

        preview = background.copy()
        yolo_lines = []

        # iterate layers
        for layer in psd:

            if not layer.is_visible():
                continue
            if self.is_numeric(layer.name):
                continue
            if layer.name == bg_name:
                continue
            if layer.name in ignored_names:
                continue

            pil_img = layer.composite()
            if pil_img is None:
                continue

            layer_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
            alpha = layer_img[:, :, 3]

            if np.max(alpha) == 0:
                continue

            _, thresh = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            cname = layer.name.strip()

            if cname not in self.class_map:
                self.class_map[cname] = len(self.class_map)

            cid = self.class_map[cname]
            x0, y0, _, _ = layer.bbox

            for cnt in contours:

                cnt = cnt + np.array([[x0, y0]])
                x, y, w, h = cv2.boundingRect(cnt)

                x, y, w, h = self.make_padded_square_bbox(
                    x, y, w, h, img_w, img_h, self.pad_ratio
                )

                xc, yc, wn, hn = self.bbox_to_yolo(x, y, w, h, img_w, img_h)

                yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview, cname, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # write outputs
        with open(os.path.join(labels_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))

        cv2.imwrite(os.path.join(preview_dir, f"{name}_boxes.png"), preview)

        print(f"Saved {len(yolo_lines)} boxes")

    # ------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------

    def run(self):

        psd_files = sorted(
            [f for f in os.listdir(self.input_folder) if f.lower().endswith(".psd")],
            key=self.natural_key
        )

        if not psd_files:
            print("No PSD files.")
            return

        # PASS 1
        layer_choices = self.collect_layer_choices(psd_files)

        # PASS 2
        print("\n========== PROCESSING ==========\n")

        for f in psd_files:
            self.process_single_psd(
                os.path.join(self.input_folder, f),
                layer_choices,
            )

        # write classes.txt
        inv = {v: k for k, v in self.class_map.items()}
        with open(os.path.join(self.output_root, "classes.txt"), "w") as f:
            for i in range(len(inv)):
                f.write(inv[i] + "\n")

        print("\nDONE. YOLO dataset ready.")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    PSDToYOLO("Images", "output_yolo", pad_ratio=0.15).run()
