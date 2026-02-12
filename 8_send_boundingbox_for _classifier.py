import os
import cv2

# -------- CONFIG --------
DATASET_ROOT = "tiled_dataset"   # root YOLO dataset
OUTPUT_DIR = "extracted_boxes"   # folder of cropped detections
# ------------------------


def process_split(split):
    img_dir = os.path.join(DATASET_ROOT, "images", split)
    lbl_dir = os.path.join(DATASET_ROOT, "labels", split)

    out_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    count = 0

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, img_name.replace(".png", ".txt"))

        if not os.path.exists(txt_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        with open(txt_path) as f:
            for i, line in enumerate(f):
                cls, x, y, bw, bh = map(float, line.split()[:5])

                # YOLO → pixel coords
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                out_name = f"{img_name[:-4]}_box{i}.png"
                cv2.imwrite(os.path.join(out_dir, out_name), crop)
                count += 1

    print(f"{split}: extracted {count} boxes → {out_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_split("train")
    process_split("val")

    print("\n✅ All bounding boxes extracted.")


if __name__ == "__main__":
    main()
