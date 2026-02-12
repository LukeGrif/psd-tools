import os
import cv2
import albumentations as A

DATASET_DIR = "tiled_dataset"
AUG_FACTOR = 5  # sensible default


# -----------------------------
# Clamp YOLO bbox to valid range
# -----------------------------

def clamp_bbox(box):
    x, y, w, h = box

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(1e-6, min(1.0, w))
    h = max(1e-6, min(1.0, h))

    return [x, y, w, h]


# -----------------------------
# Augmentation pipeline
# -----------------------------

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.7),
        A.RandomBrightnessContrast(p=0.6),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.2,
    ),
)


# -----------------------------
# Load YOLO labels
# -----------------------------

def load_labels(path):
    boxes = []
    classes = []

    if not os.path.exists(path):
        return boxes, classes

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            classes.append(int(parts[0]))
            boxes.append(clamp_bbox([float(x) for x in parts[1:]]))

    return boxes, classes


# -----------------------------
# Save YOLO labels
# -----------------------------

def save_labels(path, boxes, classes):
    with open(path, "w") as f:
        for cls, box in zip(classes, boxes):
            box = clamp_bbox(box)
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")


# -----------------------------
# Augment a single image
# -----------------------------

def augment_image(img_path, lbl_path, out_img_dir, out_lbl_dir):
    image = cv2.imread(img_path)
    if image is None:
        return

    boxes, classes = load_labels(lbl_path)

    if not boxes:
        return

    base = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(AUG_FACTOR):
        try:
            augmented = transform(image=image, bboxes=boxes, class_labels=classes)
        except Exception:
            continue

        new_img = augmented["image"]
        new_boxes = augmented["bboxes"]
        new_classes = augmented["class_labels"]

        if not new_boxes:
            continue

        img_name = f"{base}_aug{i}.png"
        lbl_name = f"{base}_aug{i}.txt"

        cv2.imwrite(os.path.join(out_img_dir, img_name), new_img)
        save_labels(os.path.join(out_lbl_dir, lbl_name), new_boxes, new_classes)


# -----------------------------
# Main augmentation loop (TRAIN ONLY)
# -----------------------------

def main():
    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")

    if not os.path.exists(train_img_dir):
        print("❌ Train directory not found:", train_img_dir)
        return

    images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(".png")]

    for img in images:
        img_path = os.path.join(train_img_dir, img)
        lbl_path = os.path.join(train_lbl_dir, img.replace(".png", ".txt"))

        if not os.path.exists(lbl_path):
            continue

        augment_image(img_path, lbl_path, train_img_dir, train_lbl_dir)

    print("✅ Augmentation complete (train set only)")


if __name__ == "__main__":
    main()
