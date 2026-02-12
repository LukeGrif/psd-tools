import os
import random
import shutil
from collections import defaultdict

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SOURCE_ROOT = "output"              # species folders created in Part 1
DEST_ROOT = "classifier_dataset"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)


# ------------------------------------------------------------
# Discover classes automatically
# ------------------------------------------------------------

CLASSES = sorted([
    d for d in os.listdir(SOURCE_ROOT)
    if os.path.isdir(os.path.join(SOURCE_ROOT, d))
])

print(f"Discovered {len(CLASSES)} classes.\n")


# ------------------------------------------------------------
# Prepare output structure
# ------------------------------------------------------------

for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, split, cls), exist_ok=True)


# ------------------------------------------------------------
# Split per class (CORRECT ML PRACTICE)
# ------------------------------------------------------------

train_counts = defaultdict(int)
val_counts = defaultdict(int)

for cls in CLASSES:

    cls_path = os.path.join(SOURCE_ROOT, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(".png")]

    if not images:
        continue

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # --- copy files ---
    for f in train_imgs:
        shutil.copy2(
            os.path.join(cls_path, f),
            os.path.join(DEST_ROOT, "train", cls, f)
        )
        train_counts[cls] += 1

    for f in val_imgs:
        shutil.copy2(
            os.path.join(cls_path, f),
            os.path.join(DEST_ROOT, "val", cls, f)
        )
        val_counts[cls] += 1


# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------

print("\nDataset built.")
print(f"Location: {DEST_ROOT}\n")

print("--- TRAIN counts ---")
for cls in CLASSES:
    print(f"{cls}: {train_counts[cls]}")

print("\n--- VAL counts ---")
for cls in CLASSES:
    print(f"{cls}: {val_counts[cls]}")

print()

for cls in CLASSES:
    total = train_counts[cls] + val_counts[cls]
    if total < 20:
        print(f"Very few samples for '{cls}' (total {total}). Results may be unstable.")
