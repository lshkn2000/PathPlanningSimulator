import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def forward(self, input, size=256 * 6 * 6):
        return input.view(input.size(0), size, 1, 1)


class ConvVAE(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size
        self.img_channels = img_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.img_channels, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            Flatten(),
        )
        # for encode Latent mean and std
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)
        self.fc_logsigma = nn.Linear(256 * 6 * 6, latent_size)

        self.fc_for_decode = nn.Linear(latent_size, 256 * 6 * 6)
        # Decoder
        # The decoder will reconstruct the image to the size of the input image in order to calculate the loss function.
        # for decode reconstruction
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=256 * 6 * 6, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(6, 6), stride=(2, 2)),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)
        return mu, logsigma

    def decode(self, z):
        z = self.fc_for_decode(z)
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logsigma):
        # sample from mean and std
        std = torch.exp(logsigma.mul(0.5))     # std = sqrt(sigma)
        eps = torch.randn_like(std)         # random noise
        return eps * std + mu

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        recon_x = self.decode(z)
        return recon_x, mu, logsigma

    def loss_function(self, recon_x, x, mu, logsigma, kld_weight):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # BCE = F.mse_loss(recon_x, x, size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - logsigma.exp())
        return BCE + KLD
