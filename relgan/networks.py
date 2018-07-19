import torch

from relgan.utils import Flatten, Reshape


def Classifier():
    return torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28**2, 512),
        torch.nn.PReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 1),
    )


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(64, 512),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 1024),
            torch.nn.PReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 28**2),
            Reshape(28, 28),
        )

    def forward(self, n):
        vec = torch.randn(n, 64)
        return self.generator(vec)
