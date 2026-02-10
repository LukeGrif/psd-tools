import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# ---------------- CONFIG ----------------
DATA_DIR = "classifier_dataset"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
PATIENCE = 5  # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # --- augmentation (train only) ---
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ---------------- DATASETS ----------------
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print("Classes:", train_dataset.classes)

# ---------------- MODEL ----------------
model = models.resnet18(weights="IMAGENET1K_V1")

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(DEVICE)

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0
epochs_no_improve = 0

train_losses = []
val_accuracies = []

# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ===== VALIDATE =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total if total > 0 else 0
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # ===== EARLY STOPPING =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0

        torch.save(model.state_dict(), "best_species_classifier.pth")
        print("✔ Best model saved")

    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered")
            break


# ---------------- FINAL SAVE ----------------
torch.save(model.state_dict(), "last_species_classifier.pth")

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("Saved:")
print(" - best_species_classifier.pth")
print(" - last_species_classifier.pth")
