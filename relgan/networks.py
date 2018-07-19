import torch

from relgan.utils import Flatten, Reshape


def Classifier():
    return torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(28**2, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.PReLU(),
        torch.nn.Linear(256, 1),
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
        vec = torch.randn(n, 64)
        return self.generator(vec)
