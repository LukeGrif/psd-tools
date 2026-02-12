import os

LABEL_DIR = "dataset_split/labels"

for split in ["train", "val", "test"]:
    folder = os.path.join(LABEL_DIR, split)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        new_lines = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, x, y, w, h = parts
                x, y, w, h = map(float, (x, y, w, h))

                # keep only valid YOLO boxes
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    new_lines.append(line.strip())

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("✅ Labels cleaned")
