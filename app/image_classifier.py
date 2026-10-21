import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import CLASSES, AssignmentCNN

WEIGHTS_PATH = Path(__file__).parent / "cnn_cifar10.pth"

# Same preprocessing used in training: resize CIFAR-10 (32x32) up to the required 64x64
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class ImageClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = AssignmentCNN()
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = TRANSFORM(image).unsqueeze(0)          # [1, 3, 64, 64]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return {"predicted_class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}
