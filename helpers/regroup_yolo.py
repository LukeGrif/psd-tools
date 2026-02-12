import os
import shutil

INPUT_DIR = "output_yolo"
OUTPUT_DIR = "output_grouped"

OLD_CLASSES = os.path.join(INPUT_DIR, "classes.txt")
OLD_LABELS = os.path.join(INPUT_DIR, "labels")
OLD_IMAGES = os.path.join(INPUT_DIR, "images")


# --------------------------------------------------
# GROUP MAP (UPDATED FOR REAL SPECIES)
# --------------------------------------------------

GROUP_MAP = {
    # 0 — Anemones & cup corals
    "Corynactis viridis": 0,
    "Caryophyllia smithii": 0,

    # 1 — Colonial tunicates & bryozoans
    "Botryllus Schlosseri": 1,
    "Dendrodoa grossularia": 1,
    "Lissoclinum perforatum": 1,
    "Bryozoa sp": 1,
    "Crisia bryozoans": 1,
    "Pentapora foliacea": 1,

    # 2 — Coralline crust
    "Coralline crust": 2,
    "Coralline crusts": 2,

    # 3 — Sponges
    "Haliclona viscosa": 3,
    "Tethya citrina": 3,

    # 4 — Echinoderms
    "Asterias rubens": 4,
    "Echinus esculentus": 4,
    "Antedon bifida": 4,

    # 5 — Mollusc & tube worm
    "Steromphala cineraria": 5,
    "Spirodbranchus triqueter": 5,
}

NEW_CLASSES = [
    "anemone_coral",
    "colonial_bryozoan_tunicate",
    "coralline_crust",
    "sponge",
    "echinoderm",
    "mollusc_worm",
]


# --------------------------------------------------
# LOAD OLD CLASS LIST
# --------------------------------------------------

with open(OLD_CLASSES) as f:
    old_names = [l.strip() for l in f.readlines()]

id_to_name = {i: n for i, n in enumerate(old_names)}


# --------------------------------------------------
# PREP OUTPUT
# --------------------------------------------------

new_labels = os.path.join(OUTPUT_DIR, "labels")
new_images = os.path.join(OUTPUT_DIR, "images")

os.makedirs(new_labels, exist_ok=True)
os.makedirs(new_images, exist_ok=True)


# copy images
for img in os.listdir(OLD_IMAGES):
    shutil.copy(os.path.join(OLD_IMAGES, img), os.path.join(new_images, img))


# --------------------------------------------------
# REMAP LABEL FILES
# --------------------------------------------------

for file in os.listdir(OLD_LABELS):

    in_path = os.path.join(OLD_LABELS, file)
    out_path = os.path.join(new_labels, file)

    new_lines = []

    with open(in_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        old_id = int(parts[0])
        name = id_to_name[old_id]

        if name not in GROUP_MAP:
            continue

        new_id = GROUP_MAP[name]
        new_lines.append(" ".join([str(new_id)] + parts[1:]))

    with open(out_path, "w") as f:
        f.write("\n".join(new_lines))


# --------------------------------------------------
# WRITE NEW classes.txt
# --------------------------------------------------

with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for c in NEW_CLASSES:
        f.write(c + "\n")


print("\nGrouped YOLO dataset ready at:", OUTPUT_DIR)
