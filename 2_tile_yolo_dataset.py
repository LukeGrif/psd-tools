import os
import cv2

DATASET_DIR = "dataset"          # your current dataset root
OUTPUT_DIR = "tiled_dataset"     # new tiled dataset

TILE_SIZE = 640
STRIDE = 320  # overlap


def load_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            boxes.append((int(cls), x, y, w, h))
    return boxes


def save_labels(path, boxes):
    with open(path, "w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_pixel(box, img_w, img_h):
    cls, x, y, w, h = box
    return (
        cls,
        (x - w / 2) * img_w,
        (y - h / 2) * img_h,
        (x + w / 2) * img_w,
        (y + h / 2) * img_h,
    )


def pixel_to_yolo(box, tile_w, tile_h):
    cls, x1, y1, x2, y2 = box
    x = ((x1 + x2) / 2) / tile_w
    y = ((y1 + y2) / 2) / tile_h
    w = (x2 - x1) / tile_w
    h = (y2 - y1) / tile_h
    return cls, x, y, w, h


def process_split(split):
    in_img_dir = os.path.join(DATASET_DIR, "images", split)
    in_lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    out_img_dir = os.path.join(OUTPUT_DIR, "images", split)
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for img_name in os.listdir(in_img_dir):

        img_path = os.path.join(in_img_dir, img_name)
        lbl_path = os.path.join(in_lbl_dir, img_name.replace(".png", ".txt"))

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        boxes = load_labels(lbl_path)
        boxes_px = [yolo_to_pixel(b, w, h) for b in boxes]

        tile_id = 0

        for y in range(0, h - TILE_SIZE + 1, STRIDE):
            for x in range(0, w - TILE_SIZE + 1, STRIDE):

                tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
                new_boxes = []

                for cls, x1, y1, x2, y2 in boxes_px:

                    nx1 = max(x1 - x, 0)
                    ny1 = max(y1 - y, 0)
                    nx2 = min(x2 - x, TILE_SIZE)
                    ny2 = min(y2 - y, TILE_SIZE)

                    if nx2 <= nx1 or ny2 <= ny1:
                        continue

                    new_boxes.append(
                        pixel_to_yolo((cls, nx1, ny1, nx2, ny2), TILE_SIZE, TILE_SIZE)
                    )

                if not new_boxes:
                    continue

                out_img = f"{img_name[:-4]}_tile{tile_id}.png"
                out_lbl = f"{img_name[:-4]}_tile{tile_id}.txt"

                cv2.imwrite(os.path.join(out_img_dir, out_img), tile)
                save_labels(os.path.join(out_lbl_dir, out_lbl), new_boxes)

                tile_id += 1


def main():
    process_split("train")
    process_split("val")
    print("✅ Tiling complete →", OUTPUT_DIR)


if __name__ == "__main__":
    main()
