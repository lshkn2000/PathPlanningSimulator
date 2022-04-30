import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256 * 2 * 2):
        return input.view(input.size(0), size, 1, 1)


class ConvVAE(nn.Module):
    def __init__(self, img_channels, latent_dim, hidden_dims:List = None):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_dim
        self.img_channels = img_channels
        self.KLD_weight = 0.00025

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(img_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            img_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # for encode Latent mean and std
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.fc_for_decode = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        # Decoder
        # The decoder will reconstruct the image to the size of the input image in order to calculate the loss function.
        # for decode reconstruction
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid()
                            # nn.Tanh()
                            )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma

    def decode(self, z):
        z = self.fc_for_decode(z)
        z = z.view(-1, 512, 2, 2)
        result = self.decoder(z)
        recon = self.final_layer(result)
        return recon

    def reparameterize(self, mu, logsigma):
        # sample from mean and std
        std = torch.exp(logsigma * 0.5)     # std = sqrt(sigma)
        eps = torch.randn_like(std)         # random noise
        z = eps * std + mu
        return z

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        recon_x = self.decode(z)
        return recon_x, mu, logsigma, z

    def loss_function(self, recon_x, x, mu, logsigma):
        # decoder 마지막에 sigmoid 사용하는 경우 (BCD는 target 과 input 값이 0~1 사이 값이어야 한다.)
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        # BCE + self.KLD_weight * KLD, BCE, KLD

        # decoder 마지막에 tanh 사용하는 경우
        # recons_loss = F.mse_loss(recon_x, x)
        # KLD = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim = 1), dim=0)
        # recons_loss + self.KLD_weight * KLD, recons_loss, KLD
        return BCE + self.KLD_weight * KLD, BCE, KLD
