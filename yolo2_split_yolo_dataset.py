import os
import shutil
import random


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

INPUT_ROOT = "output_yolo"   # folder from your previous script
OUTPUT_ROOT = "dataset"      # final YOLO dataset folder
TRAIN_RATIO = 0.8            # 80% train / 20% val
SEED = 42                    # reproducible split


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def make_dirs():
    """Create YOLO dataset folder structure."""
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)


def copy_pair(image_name, split):
    """
    Copy image + label pair into train or val folder.
    """
    src_img = os.path.join(INPUT_ROOT, "images", image_name)
    src_lbl = os.path.join(INPUT_ROOT, "labels", image_name.replace(".png", ".txt"))

    dst_img = os.path.join(OUTPUT_ROOT, "images", split, image_name)
    dst_lbl = os.path.join(OUTPUT_ROOT, "labels", split, image_name.replace(".png", ".txt"))

    shutil.copy2(src_img, dst_img)

    # label might not exist if 0 objects
    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
    else:
        # create empty label file (valid YOLO behaviour)
        open(dst_lbl, "w").close()


# ------------------------------------------------------------
# Main split logic
# ------------------------------------------------------------

def split_dataset():

    random.seed(SEED)

    image_dir = os.path.join(INPUT_ROOT, "images")
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    if not images:
        print("❌ No images found in output_yolo/images")
        return

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")

    # create folders
    make_dirs()

    # copy files
    for img in train_images:
        copy_pair(img, "train")

    for img in val_images:
        copy_pair(img, "val")

    # copy classes.txt
    shutil.copy2(
        os.path.join(INPUT_ROOT, "classes.txt"),
        os.path.join(OUTPUT_ROOT, "classes.txt"),
    )

    print("\n✅ Dataset split complete.")
    print(f"Saved to → {OUTPUT_ROOT}/")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    split_dataset()
