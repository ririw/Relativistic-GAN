"""
Relgan training harness.
"""
import torch

from relgan import experiment

batch_size = 31


class RelGAN:
    def __init__(self, generator, critic, data):
        super().__init__()
        self.data = data
        self.critic: torch.nn.Module = critic
        self.generator: torch.nn.Module = generator

        self.critic_opt = torch.optim.Adam(self.critic.parameters())
        self.generator_opt = torch.optim.Adam(self.generator.parameters())

    def step(self):
        self.d_step()
        self.g_step()

    def d_step(self):
        batch = torch.randperm(self.data.shape[0])[:batch_size]
        self.critic_opt.zero_grad()

        real_img = self.data[batch]
        gene_img = self.generator(batch_size).detach()

        y = torch.ones(batch_size)
        y_real = self.critic(real_img).view(-1)
        y_gene = self.critic(gene_img).view(-1)

        err_d = (
                torch.mean((y_real - torch.mean(y_gene) - 1) ** 2) +
                torch.mean((y_gene - torch.mean(y_real) + 1) ** 2)
        )
        err_d.backward()
        self.critic_opt.step()
        experiment.ex.log_scalar('d.loss', err_d.item())

    def g_step(self):
        batch = torch.randperm(self.data.shape[0])[:batch_size]
        self.generator_opt.zero_grad()

        real_img = self.data[batch]
        gene_img = self.generator(batch_size)

        y_real = self.critic(real_img).view(-1)
        y_gene = self.critic(gene_img).view(-1)

        err_g = (
                torch.mean((y_real - torch.mean(y_gene) + 1) ** 2) +
                torch.mean((y_gene - torch.mean(y_real) - 1) ** 2)
        )
        err_g.backward()
        self.generator_opt.step()
        experiment.ex.log_scalar('g.loss', err_g.item())
