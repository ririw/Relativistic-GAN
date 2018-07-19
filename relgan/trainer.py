"""
Relgan training harness.
"""
import torch


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
        batch = torch.randperm(self.data.shape[0])[:32]
        self.critic_opt.zero_grad()

        real_img = self.data[batch]
        gene_img = self.generator(32).detach()

        y = torch.ones(32)
        y_real = self.critic(real_img).view(-1)
        y_gene = self.critic(gene_img).view(-1)

        loss = torch.nn.BCEWithLogitsLoss()
        err_d = loss(y_real - y_gene, y)
        err_d.backward()
        self.critic_opt.step()

    def g_step(self):
        batch = torch.randperm(self.data.shape[0])[:32]
        self.generator_opt.zero_grad()

        real_img = self.data[batch]
        gene_img = self.generator(32)

        y_real = self.critic(real_img).view(-1)
        y_gene = self.critic(gene_img).view(-1)
        y = torch.ones(32)

        loss = torch.nn.BCEWithLogitsLoss()
        err_g = loss(y_real - y_gene, y)
        err_g.backward()
        self.generator_opt.step()
