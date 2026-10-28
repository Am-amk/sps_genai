import io
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from app.gan_model import Z_DIM, Generator

WEIGHTS_PATH = Path(__file__).parent / "gan_generator.pth"


class DigitGenerator:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.generator = Generator()
        self.generator.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.generator.eval()

    def generate_png(self, num_images: int = 16, seed: int | None = None) -> bytes:
        """Generate num_images digits and return them as one PNG grid."""
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(num_images, Z_DIM)
            images = self.generator(z)
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        grid = make_grid(images, nrow=min(num_images, 8), padding=2, pad_value=1.0)
        buffer = io.BytesIO()
        save_image(grid, buffer, format="png")
        return buffer.getvalue()
