import torch

from relgan.utils import Flatten, Reshape


def Classifier():
    return torch.nn.Sequential(
        Reshape(1, 28, 28),
        torch.nn.Conv2d(1, 32, 3),
        torch.nn.MaxPool2d(2),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, 64, 5),
        torch.nn.BatchNorm2d(64),
        Flatten(),
        torch.nn.Linear(5184, 512),
        torch.nn.PReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.PReLU(),
        torch.nn.Linear(128, 1),
    )


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 128),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1024),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 28**2),
            torch.nn.Sigmoid(),
            Reshape(28, 28),
        )

    def forward(self, n):
        device = next(self.generator.parameters()).device
        vec = torch.randn(n, 64).to(device)
        return self.generator(vec)
