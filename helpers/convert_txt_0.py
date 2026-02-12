import os
import glob

folder = "/home/luke-griffin/psd-tools/output_yolo/labels"

# Find all .txt files in the folder
txt_files = glob.glob(os.path.join(folder, "*.txt"))

for file_path in txt_files:
    new_lines = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            # Skip empty lines
            if not parts:
                continue

            # Set first number to 0
            parts[0] = "0"

            new_lines.append(" ".join(parts))

    # Overwrite the original file
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print(f"Done. Processed {len(txt_files)} files.")
