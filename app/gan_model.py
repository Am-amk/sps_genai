import torch
import torch.nn as nn

Z_DIM = 100


class Generator(nn.Module):
    """Assignment 3 generator: noise (B, 100) -> image (B, 1, 28, 28) in [-1, 1]."""

    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """Assignment 3 discriminator: image (B, 1, 28, 28) -> real/fake probability (B, 1)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28x28 -> 14x14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
