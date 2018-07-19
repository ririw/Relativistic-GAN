import torch


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = [-1] + list(shape)

    def forward(self, x):
        res = x.view(*self.shape)
        msg = 'Incorrect shape size:\n\tinpt: {}\n\toutput: {}'.format(res.shape, x.shape)
        assert res.shape[0] == x.shape[0], msg
        return res