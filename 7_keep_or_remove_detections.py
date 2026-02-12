import os
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
DATASET_ROOT = "yolo_dataset_from_test"  # existing YOLO dataset to review
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
# ----------------------------------------


def load_dataset_detections():
    """Load existing YOLO labels and convert to pixel boxes for review."""

    detections = []

    for img_name in sorted(os.listdir(IMAGES_DIR)):
        img_path = os.path.join(IMAGES_DIR, img_name)
        txt_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(txt_path):
            continue

        h, w = image.shape[:2]

        with open(txt_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls, x, y, bw, bh, conf = parts

                # Convert YOLO → pixel xyxy
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                detections.append({
                    "image_path": img_path,
                    "label_path": txt_path,
                    "image": image,
                    "bbox": (x1, y1, x2, y2),
                    "cls": int(cls),
                    "conf": float(conf),
                })

    return detections


def save_image_labels(image_path, kept_dets):
    """Rewrite label file for a single image in real time."""

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    txt_path = os.path.join(LABELS_DIR, os.path.basename(image_path).replace(".jpg", ".txt"))

    with open(txt_path, "w") as f:
        for d in kept_dets:
            x1, y1, x2, y2 = d["bbox"]

            # Convert pixel → YOLO
            x = ((x1 + x2) / 2) / w
            y = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"{d['cls']} {x} {y} {bw} {bh} {d['conf']}\n")


def review_dataset(detections):
    """Interactive reviewer that updates labels in real time."""

    idx = 0
    kept_by_image = {}

    while 0 <= idx < len(detections):
        det = detections[idx]
        img = det["image"].copy()
        x1, y1, x2, y2 = det["bbox"]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"cls={det['cls']} conf={det['conf']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Dataset Reviewer", img)
        key = cv2.waitKey(0)

        img_key = det["image_path"]
        kept_by_image.setdefault(img_key, [])

        # k = keep
        if key == ord('k'):
            kept_by_image[img_key].append(det)
            save_image_labels(img_key, kept_by_image[img_key])
            idx += 1

        # d = delete
        elif key == ord('d'):
            save_image_labels(img_key, kept_by_image[img_key])
            idx += 1

        # a = previous
        elif key == ord('a'):
            idx -= 1

        # q = quit
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    print("Loading existing YOLO dataset...")
    detections = load_dataset_detections()

    print(f"Total detections to review: {len(detections)}")
    print("\nControls:")
    print("  k = keep (writes immediately)")
    print("  d = delete (updates immediately)")
    print("  a = previous")
    print("  q = quit\n")

    review_dataset(detections)

    print("\n✅ Review complete. Dataset updated in place.")


if __name__ == "__main__":
    main()
