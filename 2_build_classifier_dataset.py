import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

INPUT_DIR = "output"                # crops from PSD extractor
OUTPUT_DIR = "classifier_dataset"   # final dataset
IMAGE_SIZE = 224                    # CNN input size
TRAIN_SPLIT = 0.8                   # split by PSD folder (group split)
RANDOM_SEED = 42                    # reproducibility

# Only keep these classes (must match the filename prefix before "_obj")
KEEP_CLASSES = {
    "Corynactis viridis",
    "Haliclona viscosa",
}

# Optional: cap per class per split to reduce imbalance (None = no cap)
MAX_PER_CLASS_PER_SPLIT = None  # e.g. 500


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_class_name(filename: str) -> str:
    """
    From:
        "Corynactis viridis_obj3.png"
    → "Corynactis viridis"
    """
    return filename.split("_obj")[0]


def safe_dir_name(class_name: str) -> str:
    """Folder-safe name (keeps readability)."""
    return class_name.replace("/", "_").strip()


def resize_and_save(in_path: Path, out_path: Path, size: int) -> bool:
    """Resize to fixed square and save. Returns True if saved."""
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), resized)


# ------------------------------------------------------------
# Main dataset builder
# ------------------------------------------------------------

def build_classifier_dataset():
    random.seed(RANDOM_SEED)

    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)

    if not input_root.exists():
        print(f"INPUT_DIR not found: {INPUT_DIR}")
        return

    # clean output
    if output_root.exists():
        shutil.rmtree(output_root)

    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "val").mkdir(parents=True, exist_ok=True)

    # All PSD groups (folders)
    psd_folders = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not psd_folders:
        print("No PSD output folders found in INPUT_DIR.")
        return

    # --- Filter groups to only those that contain at least 1 image from KEEP_CLASSES ---
    def group_has_kept_class(folder: Path) -> bool:
        for img_path in folder.glob("*.png"):
            if extract_class_name(img_path.name) in KEEP_CLASSES:
                return True
        return False

    psd_folders = [p for p in psd_folders if group_has_kept_class(p)]
    if not psd_folders:
        print("No PSD folders contain any of the selected classes.")
        return

    print(f"Found {len(psd_folders)} PSD groups containing selected classes.")

    # deterministic shuffle, then split by group
    random.shuffle(psd_folders)
    split_index = int(len(psd_folders) * TRAIN_SPLIT)

    train_groups = psd_folders[:split_index]
    val_groups = psd_folders[split_index:]

    print(f"Train PSDs: {len(train_groups)}")
    print(f"Val PSDs:   {len(val_groups)}")

    # track counts
    counts = {
        "train": defaultdict(int),
        "val": defaultdict(int)
    }

    # optional per-class caps
    caps = {
        "train": defaultdict(lambda: MAX_PER_CLASS_PER_SPLIT),
        "val": defaultdict(lambda: MAX_PER_CLASS_PER_SPLIT)
    }

    def process_groups(groups, split_name: str):
        # to keep reproducibility but avoid order bias, sort images per folder
        for folder in sorted(groups):
            images = sorted(folder.glob("*.png"))

            # shuffle images deterministically per folder (based on seed + folder name)
            rng = random.Random((RANDOM_SEED, folder.name, split_name))
            rng.shuffle(images)

            for img_path in images:
                class_name = extract_class_name(img_path.name)
                if class_name not in KEEP_CLASSES:
                    continue

                # cap handling
                cap = MAX_PER_CLASS_PER_SPLIT
                if cap is not None and counts[split_name][class_name] >= cap:
                    continue

                out_class_dir = output_root / split_name / safe_dir_name(class_name)
                out_path = out_class_dir / img_path.name

                saved = resize_and_save(img_path, out_path, IMAGE_SIZE)
                if saved:
                    counts[split_name][class_name] += 1

    process_groups(train_groups, "train")
    process_groups(val_groups, "val")

    print("\nDataset built.")
    print(f"Location: {OUTPUT_DIR}\n")

    # summary
    for split in ["train", "val"]:
        print(f"--- {split.upper()} counts ---")
        for cls in sorted(KEEP_CLASSES):
            print(f"{cls}: {counts[split][cls]}")
        print()

    # warn if very low counts
    for cls in sorted(KEEP_CLASSES):
        total = counts["train"][cls] + counts["val"][cls]
        if total < 10:
            print(f"Very few samples for '{cls}' (total {total}). Results may be unstable.")


if __name__ == "__main__":
    build_classifier_dataset()
