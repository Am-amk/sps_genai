import torch
import torch.nn as nn

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class AssignmentCNN(nn.Module):
    """CNN from Assignment 2: input 3 x 64 x 64, output 10 class scores."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   # 16 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 32 x 16 x 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # 8192
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),                            # raw scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
