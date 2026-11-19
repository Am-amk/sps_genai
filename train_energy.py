import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from app.energy_model import EnergyModel, generate_samples


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
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


# -----------------------------
# Replay Buffer
# -----------------------------
class Buffer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.examples = [
            torch.rand((1, 3, 32, 32), device=device) * 2 - 1
            for _ in range(128)
        ]

    def sample_new_examples(
        self,
        batch_size,
        steps=60,
        step_size=10.0,
        noise=0.005
    ):
        n_new = np.random.binomial(batch_size, 0.05)

        new_imgs = torch.rand(
            (n_new, 3, 32, 32),
            device=self.device
        ) * 2 - 1

        old_imgs = torch.cat(
            random.choices(
                self.examples,
                k=batch_size - n_new
            ),
            dim=0
        )

        inp_imgs = torch.cat(
            [new_imgs, old_imgs],
            dim=0
        )

        new_samples = generate_samples(
            self.model,
            inp_imgs,
            steps=steps,
            step_size=step_size,
            noise_std=noise
        )

        self.examples = (
            list(torch.split(new_samples, 1, dim=0))
            + self.examples
        )

        self.examples = self.examples[:8192]

        return new_samples


# -----------------------------
# Model
# -----------------------------
model = EnergyModel().to(device)

buffer = Buffer(model, device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.0, 0.999)
)


# -----------------------------
# Smoke-test training
# -----------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    running_real = 0.0
    running_fake = 0.0

    for batch_index, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)

        real_imgs = real_imgs + torch.randn_like(real_imgs) * 0.005
        real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

        fake_imgs = buffer.sample_new_examples(
            batch_size=real_imgs.size(0),
            steps=60,
            step_size=10.0,
            noise=0.005
        )

        inp_imgs = torch.cat(
            [real_imgs, fake_imgs],
            dim=0
        )

        scores = model(inp_imgs)

        real_out, fake_out = torch.split(
            scores,
            real_imgs.size(0),
            dim=0
        )

        cdiv_loss = (
            real_out.mean()
            - fake_out.mean()
        )

        reg_loss = 0.1 * (
            real_out.pow(2).mean()
            + fake_out.pow(2).mean()
        )

        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1
        )

        optimizer.step()

        running_loss += loss.item()
        running_real += real_out.mean().item()
        running_fake += fake_out.mean().item()

    num_batches = len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / num_batches:.4f} | "
        f"Real energy: {running_real / num_batches:.4f} | "
        f"Fake energy: {running_fake / num_batches:.4f}"
    )

    torch.save(
        model.state_dict(),
        f"app/energy_cifar10_epoch_{epoch + 1}.pth"
    )

torch.save(
    model.state_dict(),
    "app/energy_cifar10.pth"
)

print("Energy model training complete.")
print("Saved final model to app/energy_cifar10.pth")