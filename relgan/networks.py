import torch
from relgan.utils import Flatten, Reshape


def Classifier():
    return torch.nn.Sequential(
        Reshape(1, 28, 28),
        torch.nn.Conv2d(1, 16, 3),
        torch.nn.PReLU(),
        torch.nn.Conv2d(16, 32, 5),
        torch.nn.PReLU(),
        torch.nn.MaxPool2d(2),

        torch.nn.Conv2d(32, 64, 7),
        torch.nn.PReLU(),
        torch.nn.MaxPool2d(2),

        Flatten(),
        torch.nn.Linear(256, 64),
        torch.nn.PReLU(),
        torch.nn.Linear(64, 1),
    )


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.PReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 28**2),
            Reshape(28, 28),
        )

    def forward(self, n):
        vec = torch.randn(n, 32)
        return self.generator(vec)
