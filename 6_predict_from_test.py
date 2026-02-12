from ultralytics import YOLO
import os
import shutil

MODEL_PATH = "runs/detect/clean_labels2/weights/best.pt"
SOURCE_DIR = "psd_tiled_output_to_test/images"

OUTPUT_ROOT = "yolo_dataset_from_test"
IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model.predict(
    source=SOURCE_DIR,
    imgsz=640,
    conf=0.05,
    save=False,
    verbose=False
)

for r in results:
    if r.boxes is None:
        continue

    img_name = os.path.basename(r.path)

    # --- copy image into YOLO images folder ---
    dst_img_path = os.path.join(IMAGES_DIR, img_name)
    shutil.copy(r.path, dst_img_path)

    # --- create matching label file ---
    txt_path = os.path.join(
        LABELS_DIR,
        img_name.replace(".jpg", ".txt")
    )

    with open(txt_path, "w") as f:
        for box, cls, conf in zip(
            r.boxes.xywhn,   # normalized YOLO coords
            r.boxes.cls,
            r.boxes.conf
        ):
            x, y, w, h = box.tolist()
            f.write(f"{int(cls)} {x} {y} {w} {h} {conf}\n")

print(f"✅ YOLO dataset created at: {OUTPUT_ROOT}")
