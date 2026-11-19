import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)

def generate_samples(model, inp_imgs, steps=60, step_size=10.0, noise_std=0.005):
    model.eval()

    # Freeze model parameters during sampling
    for p in model.parameters():
        p.requires_grad_(False)

    # We want gradients with respect to the input image
    inp_imgs = inp_imgs.detach().requires_grad_(True)

    for _ in range(steps):
        # Add a small amount of random noise
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        # Compute energy for current images
        energy = model(inp_imgs)

        # Gradient of energy with respect to input pixels
        grads, = torch.autograd.grad(
            energy,
            inp_imgs,
            grad_outputs=torch.ones_like(energy)
        )

        # Move image toward lower-energy region
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    # Re-enable model gradients for later training
    for p in model.parameters():
        p.requires_grad_(True)

    return inp_imgs.detach()
if __name__ == "__main__":
    model = EnergyModel()

    x = torch.rand(8, 3, 32, 32) * 2 - 1
    y = model(x)

    print("Input:", x.shape)
    print("Energy:", y.shape)

    samples = generate_samples(
        model,
        x,
        steps=5,
        step_size=1.0,
        noise_std=0.01
    )

    print("Samples:", samples.shape)