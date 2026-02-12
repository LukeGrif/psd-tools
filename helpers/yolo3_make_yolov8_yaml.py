import os

DATASET_ROOT = "dataset"
CLASSES_FILE = os.path.join(DATASET_ROOT, "classes.txt")
YAML_FILE = os.path.join(DATASET_ROOT, "dataset.yaml")


def main():

    if not os.path.exists(CLASSES_FILE):
        print("classes.txt not found in dataset/")
        return

    with open(CLASSES_FILE, "r") as f:
        classes = [c.strip() for c in f.readlines() if c.strip()]

    with open(YAML_FILE, "w") as f:
        f.write(f"path: {DATASET_ROOT}\n\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")

        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")

    print("✅ YOLOv8 dataset.yaml created.")
    print(f"Classes: {len(classes)}")


if __name__ == "__main__":
    main()
