"""Train the Assignment 2 CNN on CIFAR-10 and save weights to app/cnn_cifar10.pth.

Run with:  uv run python train_cnn.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import AssignmentCNN
from app.image_classifier import TRANSFORM, WEIGHTS_PATH

EPOCHS = 5
BATCH_SIZE = 64

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"Using device: {device}")

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=TRANSFORM)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=TRANSFORM)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = AssignmentCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Epoch {epoch + 1}/{EPOCHS}: loss={running_loss / len(train_loader):.4f}, "
          f"train acc={100 * correct / total:.2f}%")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"Saved weights to {WEIGHTS_PATH}")
