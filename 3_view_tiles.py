import os
import cv2

DATASET_DIR = "tiled_dataset"
WINDOW_NAME = "YOLO Tile Reviewer"
CURRENT_BOXES = []
DRAWING = False
START_PT = None


# -----------------------------
# Load YOLO labels
# -----------------------------

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


# -----------------------------
# Save YOLO labels
# -----------------------------

def save_labels(label_path, boxes, img_w, img_h):
    with open(label_path, "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            x = ((x1 + x2) / 2) / img_w
            y = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# -----------------------------
# Draw YOLO boxes
# -----------------------------

def draw_boxes(image, boxes):
    for cls, x1, y1, x2, y2 in boxes:
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


# -----------------------------
# Mouse callback for drawing boxes
# -----------------------------

def mouse_callback(event, x, y, flags, param):
    global DRAWING, START_PT, CURRENT_BOXES

    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
        START_PT = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        DRAWING = False
        x1, y1 = START_PT
        x2, y2 = x, y

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            CURRENT_BOXES.append((0, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))


# -----------------------------
# Review + edit images interactively
# -----------------------------

def review_split(split="train"):
    global CURRENT_BOXES

    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    if not os.path.exists(img_dir):
        print("❌ Missing directory:", img_dir)
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])

    idx = 0
    deleted_count = 0

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        if len(images) == 0:
            print("No images left in this split.")
            break

        idx = max(0, min(idx, len(images) - 1))

        img_name = images[idx]
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".png", ".txt"))

        image = cv2.imread(img_path)
        if image is None:
            idx += 1
            continue

        h, w = image.shape[:2]

        # Load existing boxes once per image
        if not CURRENT_BOXES:
            yolo_boxes = load_labels(lbl_path)
            CURRENT_BOXES = []
            for cls, x, y, bw, bh in yolo_boxes:
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                CURRENT_BOXES.append((cls, x1, y1, x2, y2))

        display = image.copy()
        display = draw_boxes(display, CURRENT_BOXES)

        cv2.putText(
            display,
            f"{split} | {idx+1}/{len(images)} | deleted: {deleted_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        # ESC → quit
        if key == 27:
            break

        # D or → → next (save first)
        elif key in [ord("d"), 83]:
            save_labels(lbl_path, CURRENT_BOXES, w, h)
            CURRENT_BOXES = []
            idx += 1

        # A or ← → previous (save first)
        elif key in [ord("a"), 81]:
            save_labels(lbl_path, CURRENT_BOXES, w, h)
            CURRENT_BOXES = []
            idx -= 1

        # X → delete image + label
        elif key == ord("x"):
            try:
                os.remove(img_path)
                if os.path.exists(lbl_path):
                    os.remove(lbl_path)

                print(f"🗑 Deleted: {img_name}")

                images.pop(idx)
                CURRENT_BOXES = []
                deleted_count += 1

            except Exception as e:
                print("Delete failed:", e)

        # Z → undo last box
        elif key == ord("z"):
            if CURRENT_BOXES:
                CURRENT_BOXES.pop()

        # R → remove ALL boxes
        elif key == ord("r"):
            CURRENT_BOXES = []

        # C → clear boxes (alias of R for safety)
        elif key == ord("c"):
            CURRENT_BOXES = []

    cv2.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------

def main():
    print("Controls:")
    print("  Mouse drag : draw new box (class 0)")
    print("  D / →      : next image (save labels)")
    print("  A / ←      : previous image (save labels)")
    print("  Z          : undo last box")
    print("  R          : remove ALL boxes")
    print("  X          : delete image + label")
    print("  ESC        : quit")

    review_split("train")


if __name__ == "__main__":
    main()
