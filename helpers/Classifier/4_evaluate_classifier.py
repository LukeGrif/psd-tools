import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np


# ---------------- CONFIG ----------------
DATA_DIR = "classifier_dataset"
MODEL_PATH = "species_classifier.pth"
WRONG_DIR = "misclassified"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)


# ---------------- MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


# ---------------- PREP OUTPUT FOLDER ----------------
if os.path.exists(WRONG_DIR):
    shutil.rmtree(WRONG_DIR)

for cls in class_names:
    os.makedirs(os.path.join(WRONG_DIR, cls), exist_ok=True)


# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())


# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
print(f"\nValidation Accuracy: {accuracy:.2f}%\n")


# ---------------- SAVE MISCLASSIFIED IMAGES ----------------
print("Saving misclassified images...")

for idx, (pred, label) in enumerate(zip(all_preds, all_labels)):
    if pred != label:
        img_path, _ = val_dataset.samples[idx]

        true_name = class_names[label]
        pred_name = class_names[pred]

        # filename shows true → predicted
        filename = f"{os.path.basename(img_path)}__TRUE_{true_name}__PRED_{pred_name}.png"

        dest = os.path.join(WRONG_DIR, true_name, filename)
        shutil.copy(img_path, dest)

print(f"Done. Misclassified images saved in: {WRONG_DIR}")
