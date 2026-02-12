import os
import cv2

DATASET_DIR = "tiled_dataset"  # augmented tiles live in the same train folder
WINDOW_NAME = "Augmented Dataset Viewer"


def load_labels(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x, y, w, h = parts
            boxes.append((int(cls), float(x), float(y), float(w), float(h)))

    return boxes


def draw_boxes(image, boxes):
    h, w = image.shape[:2]

    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{cls}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def view_split(split="train"):
    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    if not os.path.exists(img_dir):
        print("❌ Missing directory:", img_dir)
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])

    idx = 0

    while True:
        if idx < 0:
            idx = 0
        if idx >= len(images):
            idx = len(images) - 1

        img_name = images[idx]
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".png", ".txt"))

        image = cv2.imread(img_path)
        if image is None:
            idx += 1
            continue

        boxes = load_labels(lbl_path)
        image = draw_boxes(image, boxes)

        display = image.copy()
        cv2.putText(
            display,
            f"{split} | {idx+1}/{len(images)} | {img_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key in [ord("d"), 83]:  # next
            idx += 1
        elif key in [ord("a"), 81]:  # previous
            idx -= 1

    cv2.destroyAllWindows()


def main():
    print("Controls:")
    print("  D or → : next image")
    print("  A or ← : previous image")
    print("  ESC    : quit")

    view_split("train")


if __name__ == "__main__":
    main()
