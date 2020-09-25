import torch.nn as nn
import numpy as np
import torch
from torchsummaryX import summary

'''Conditional WGAN with gradient penalty'''


class Generator(nn.Module):
    def __init__(self, seq_size, class_dim, latent_dim):
        super(Generator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.class_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, self.seq_size),
            nn.Sigmoid()
        )

    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        trace = self.model(input)
        return trace


class Discriminator(nn.Module):
    def __init__(self, seq_size, class_dim):
        super(Discriminator, self).__init__()
        self.seq_size = seq_size
        self.class_dim = class_dim
        self.model = nn.Sequential(
            nn.Linear(self.seq_size + self.class_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, trace, c):
        input = torch.cat([trace, c], 1)
        validity = self.model(input)
        return validity
