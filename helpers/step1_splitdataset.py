import os
import shutil
import random

INPUT_DIR = "output_grouped"
OUTPUT_DIR = "dataset_split"

TRAIN = 0.7
VAL = 0.2
TEST = 0.1

random.seed(42)


def make_dirs(base):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split), exist_ok=True)


def main():
    images_dir = os.path.join(INPUT_DIR, "images")
    labels_dir = os.path.join(INPUT_DIR, "labels")

    images = sorted(os.listdir(images_dir))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN)
    n_val = int(n * VAL)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    make_dirs(OUTPUT_DIR)

    for split, files in splits.items():
        for img in files:
            label = img.replace(".png", ".txt")

            shutil.copy(
                os.path.join(images_dir, img),
                os.path.join(OUTPUT_DIR, "images", split, img),
            )

            shutil.copy(
                os.path.join(labels_dir, label),
                os.path.join(OUTPUT_DIR, "labels", split, label),
            )

    # copy classes.txt
    shutil.copy(
        os.path.join(INPUT_DIR, "classes.txt"),
        os.path.join(OUTPUT_DIR, "classes.txt"),
    )

    print("✅ Dataset split complete →", OUTPUT_DIR)


if __name__ == "__main__":
    main()
