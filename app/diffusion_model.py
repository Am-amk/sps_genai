import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Sinusoidal timestep embedding
# -----------------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies
            )
        )

        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x):
        x = x.expand(
            -1,
            1,
            1,
            self.num_frequencies
        )

        sin_part = torch.sin(
            self.angular_speeds * x
        )

        cos_part = torch.cos(
            self.angular_speeds * x
        )

        return torch.cat(
            [sin_part, cos_part],
            dim=-1
        )


# -----------------------------
# Residual block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        if in_channels != out_channels:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.proj = nn.Identity()

        self.norm = nn.BatchNorm2d(
            in_channels,
            affine=False
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        residual = self.proj(x)

        x = self.norm(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


# -----------------------------
# Downsampling block
# -----------------------------
class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(in_channels, width)
            )
            in_channels = width

        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)

        x = self.pool(x)

        return x


# -----------------------------
# Upsampling block
# -----------------------------
class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()

        self.blocks = nn.ModuleList()

        for _ in range(block_depth):
            self.blocks.append(
                ResidualBlock(
                    in_channels + width,
                    width
                )
            )

            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        for block in self.blocks:
            skip = skips.pop()

            x = torch.cat(
                [x, skip],
                dim=1
            )

            x = block(x)

        return x


# -----------------------------
# UNet
# -----------------------------
class UNet(nn.Module):
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()

        self.initial = nn.Conv2d(
            num_channels,
            32,
            kernel_size=1
        )

        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        self.embedding = SinusoidalEmbedding(
            num_frequencies=16
        )

        self.down1 = DownBlock(
            32,
            in_channels=64,
            block_depth=2
        )

        self.down2 = DownBlock(
            64,
            in_channels=32,
            block_depth=2
        )

        self.down3 = DownBlock(
            96,
            in_channels=64,
            block_depth=2
        )

        self.mid1 = ResidualBlock(
            in_channels=96,
            out_channels=128
        )

        self.mid2 = ResidualBlock(
            in_channels=128,
            out_channels=128
        )

        self.up1 = UpBlock(
            96,
            in_channels=128,
            block_depth=2
        )

        self.up2 = UpBlock(
            64,
            in_channels=96,
            block_depth=2
        )

        self.up3 = UpBlock(
            32,
            in_channels=64,
            block_depth=2
        )

        self.final = nn.Conv2d(
            32,
            num_channels,
            kernel_size=1
        )

        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []

        x = self.initial(noisy_images)

        noise_emb = self.embedding(
            noise_variances
        )

        noise_emb = F.interpolate(
            noise_emb.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size),
            mode="nearest"
        )

        x = torch.cat(
            [x, noise_emb],
            dim=1
        )

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)

# -----------------------------
# Diffusion schedule
# -----------------------------
def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95
):
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )

    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device
        )
    )

    diffusion_angles = (
        start_angle
        + diffusion_times_flat
        * (end_angle - start_angle)
    )

    signal_rates = torch.cos(
        diffusion_angles
    ).reshape(original_shape)

    noise_rates = torch.sin(
        diffusion_angles
    ).reshape(original_shape)

    return noise_rates, signal_rates


# -----------------------------
# Diffusion Model
# -----------------------------
class DiffusionModel(nn.Module):
    def __init__(self, model, schedule_fn):
        super().__init__()

        self.network = model

        self.ema_network = copy.deepcopy(
            model
        )

        self.ema_network.eval()

        self.ema_decay = 0.8

        self.schedule_fn = schedule_fn

        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def to(self, device):
        super().to(device)

        self.ema_network.to(device)

        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(
            x * self.normalizer_std
            + self.normalizer_mean,
            0.0,
            1.0
        )

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        training
    ):
        if training:
            network = self.network
            network.train()

        else:
            network = self.ema_network
            network.eval()

        pred_noises = network(
            noisy_images,
            noise_rates ** 2
        )

        pred_images = (
            noisy_images
            - noise_rates * pred_noises
        ) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps
    ):
        step_size = 1.0 / diffusion_steps

        current_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones(
                (
                    initial_noise.shape[0],
                    1,
                    1,
                    1
                ),
                device=initial_noise.device
            ) * (
                1 - step * step_size
            )

            noise_rates, signal_rates = (
                self.schedule_fn(t)
            )

            pred_noises, pred_images = (
                self.denoise(
                    current_images,
                    noise_rates,
                    signal_rates,
                    training=False
                )
            )

            next_diffusion_times = (
                t - step_size
            )

            (
                next_noise_rates,
                next_signal_rates
            ) = self.schedule_fn(
                next_diffusion_times
            )

            current_images = (
                next_signal_rates
                * pred_images
                + next_noise_rates
                * pred_noises
            )

        return pred_images

    def generate(
        self,
        num_images,
        diffusion_steps,
        image_size=32,
        initial_noise=None
    ):
        if initial_noise is None:
            initial_noise = torch.randn(
                (
                    num_images,
                    self.network.num_channels,
                    image_size,
                    image_size
                ),
                device=next(
                    self.parameters()
                ).device
            )

        with torch.no_grad():
            return self.denormalize(
                self.reverse_diffusion(
                    initial_noise,
                    diffusion_steps
                )
            )
    def train_step(self, images, optimizer, loss_fn):
        # Normalize images
        images = (
            images - self.normalizer_mean
        ) / self.normalizer_std

        # Sample random Gaussian noise
        noises = torch.randn_like(images)

        # Sample random diffusion times
        diffusion_times = torch.rand(
            (
                images.size(0),
                1,
                1,
                1
            ),
            device=images.device
        )

        # Get signal and noise rates
        noise_rates, signal_rates = (
            self.schedule_fn(
                diffusion_times
            )
        )

        # Add noise to images
        noisy_images = (
            signal_rates * images
            + noise_rates * noises
        )

        # Predict the noise
        pred_noises, _ = self.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=True
        )

        # Compare predicted noise to true noise
        loss = loss_fn(
            pred_noises,
            noises
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Update EMA network
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_network.parameters(),
                self.network.parameters()
            ):
                ema_param.copy_(
                    self.ema_decay * ema_param
                    + (1.0 - self.ema_decay) * param
                )

        return loss.item()
# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    model = UNet(
        image_size=32,
        num_channels=3
    )

    noisy_images = torch.randn(
        8,
        3,
        32,
        32
    )

    noise_variances = torch.rand(
        8,
        1,
        1,
        1
    )

    output = model(
        noisy_images,
        noise_variances
    )

    print("Input:", noisy_images.shape)
    print("Output:", output.shape)