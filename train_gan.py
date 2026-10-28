"""Train the Assignment 3 GAN on MNIST and save the generator to app/gan_generator.pth.

Run with:  uv run python train_gan.py
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from app.digit_generator import WEIGHTS_PATH
from app.gan_model import Z_DIM, Discriminator, Generator

EPOCHS = 10
BATCH_SIZE = 128
LR = 2e-4

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
print(f"Using device: {device}")

# Scale pixels to [-1, 1] to match the generator's Tanh output
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, Z_DIM, device=device)

for epoch in range(EPOCHS):
    G.train(); D.train()
    d_total, g_total = 0.0, 0.0
    for real, _ in loader:
        real = real.to(device)
        b = real.size(0)
        ones = torch.ones(b, 1, device=device)
        zeros = torch.zeros(b, 1, device=device)

        # --- Train discriminator: real -> 1, fake -> 0 ---
        fake = G(torch.randn(b, Z_DIM, device=device))
        d_loss = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # --- Train generator: make D output 1 on fakes ---
        g_loss = criterion(D(fake), ones)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}: D loss={d_total / len(loader):.4f}, "
          f"G loss={g_total / len(loader):.4f}")

G.eval()
with torch.no_grad():
    samples = (G(fixed_noise) + 1) / 2
save_image(samples, "gan_samples.png", nrow=8)
print("Saved sample grid to gan_samples.png")

torch.save(G.cpu().state_dict(), WEIGHTS_PATH)
print(f"Saved generator weights to {WEIGHTS_PATH}")
