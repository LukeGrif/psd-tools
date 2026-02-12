import os, shutil, random

SRC = "tiled_dataset"
DST = "tiles_split"

os.makedirs(f"{DST}/images/train", exist_ok=True)
os.makedirs(f"{DST}/images/val", exist_ok=True)
os.makedirs(f"{DST}/labels/train", exist_ok=True)
os.makedirs(f"{DST}/labels/val", exist_ok=True)

files = os.listdir(f"{SRC}/images")
random.shuffle(files)

split = int(len(files) * 0.8)

for i, f in enumerate(files):
    subset = "train" if i < split else "val"

    shutil.copy(f"{SRC}/images/{f}", f"{DST}/images/{subset}/{f}")
    shutil.copy(
        f"{SRC}/labels/{f.replace('.png','.txt')}",
        f"{DST}/labels/{subset}/{f.replace('.png','.txt')}",
    )

print("done")
