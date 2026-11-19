import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.diffusion_model import (
    UNet,
    DiffusionModel,
    offset_cosine_diffusion_schedule
)


# -----------------------------
# Device
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# CIFAR-10
# -----------------------------
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)


# -----------------------------
# Model
# -----------------------------
unet = UNet(
    image_size=32,
    num_channels=3,
    embedding_dim=32
)

diffusion_model = DiffusionModel(
    unet,
    offset_cosine_diffusion_schedule
)

diffusion_model.to(device)


# -----------------------------
# Normalization statistics
# -----------------------------
mean = torch.zeros(3)
std = torch.zeros(3)
total_samples = 0

stats_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

for images, _ in stats_loader:
    batch_size = images.size(0)

    images_flat = images.view(
        batch_size,
        3,
        -1
    )

    batch_mean = images_flat.mean(
        dim=(0, 2)
    )

    batch_std = images_flat.std(
        dim=(0, 2)
    )

    mean += batch_mean * batch_size
    std += batch_std * batch_size

    total_samples += batch_size

mean /= total_samples
std /= total_samples

print("Mean:", mean)
print("Std:", std)

mean = mean.reshape(
    1,
    3,
    1,
    1
).to(device)

std = std.reshape(
    1,
    3,
    1,
    1
).to(device)

diffusion_model.set_normalizer(
    mean,
    std
)


# -----------------------------
# Optimizer + loss
# -----------------------------
optimizer = torch.optim.AdamW(
    diffusion_model.network.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

loss_fn = nn.L1Loss()


# -----------------------------
# Training
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for batch_index, (images, _) in enumerate(
        train_loader
    ):
        images = images.to(device)

        loss = diffusion_model.train_step(
            images,
            optimizer,
            loss_fn
        )

        epoch_loss += loss

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss:.4f}"
            )

    avg_loss = (
        epoch_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Average Loss: {avg_loss:.4f}"
    )

    checkpoint = {
        "network": diffusion_model.network.state_dict(),
        "ema_network": diffusion_model.ema_network.state_dict(),
        "normalizer_mean": diffusion_model.normalizer_mean,
        "normalizer_std": diffusion_model.normalizer_std
    }

    torch.save(
        checkpoint,
        f"app/diffusion_cifar10_epoch_{epoch + 1}.pth"
    )


# -----------------------------
# Final save
# -----------------------------
torch.save(
    {
        "network": diffusion_model.network.state_dict(),
        "ema_network": diffusion_model.ema_network.state_dict(),
        "normalizer_mean": diffusion_model.normalizer_mean,
        "normalizer_std": diffusion_model.normalizer_std
    },
    "app/diffusion_cifar10.pth"
)

print("Diffusion model training complete.")
print("Saved final model to app/diffusion_cifar10.pth")