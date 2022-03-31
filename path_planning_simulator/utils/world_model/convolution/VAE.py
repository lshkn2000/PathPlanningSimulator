import os
import random
from typing import List
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.preprocessing import normalize, Normalizer

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


import pytorch_lightning as pl


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class VanillaVAE(nn.Module):
    def __init__(self,
                 input_dim:int,
                 latent_dim:int,
                 hidden_dim:List = None):
        super(VanillaVAE, self).__init__()

        if hidden_dim is None:
            hidden_dim = [512, 256, 128, 64, 32]

        encoder_layer = [input_dim] + hidden_dim + [latent_dim]

        # Encoder
        encoder = []
        for i in range(len(encoder_layer) - 1):
            encoder.append(
                nn.Sequential(
                    nn.Linear(encoder_layer[i], encoder_layer[i+1], bias=True),
                    nn.BatchNorm1d(encoder_layer[i + 1], momentum=0.7),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*encoder)

        # Latent
        self.latent2mu = nn.Linear(latent_dim, latent_dim)
        self.latent2log_var = nn.Linear(latent_dim, latent_dim)

        # Decoder
        decoder = []
        hidden_dim.reverse()
        decoder_layer = [latent_dim] + hidden_dim + [input_dim]
        for i in range(len(decoder_layer) - 1):
            if i < len(decoder_layer) - 2:
                decoder.append(
                    nn.Sequential(
                        nn.Linear(decoder_layer[i], decoder_layer[i+1], bias=True),
                        nn.ReLU()
                    )
                )
            else: # 마지막은 Tanh()
                decoder.append(
                    nn.Sequential(
                        nn.Linear(decoder_layer[i], decoder_layer[i+1], bias=True),
                        nn.Tanh()
                    )
                )
        self.decoder = nn.Sequential(*decoder)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.latent2mu(encoded)
        log_var = self.latent2log_var(encoded)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten in batch
        mu, log_var = self.encode(x)
        latent = self.reparameterize(mu, log_var)
        x_hat = self.decoder(latent)
        return mu, log_var, x, x_hat

    def loss_function(self, *args, kld_weight:float):
        mu = args[0]
        log_var = args[1]
        x = args[2]
        x_hat = args[3]

        # Loss
        recons_loss = F.mse_loss(x, x_hat)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_loss * kld_weight

        return loss


class VAEEXE(pl.LightningModule):
    def __init__(self, vae_model, *args, **kwargs):
        super(VAEEXE, self).__init__()

        self.model = vae_model

        # hyper parameters
        self.kld_weight = kwargs['kld_weight']
        self.learning_rate = kwargs['learning_rate']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # if x , y(label) data
        # x, y = batch # sample from replay buffer
        # if x data (without label)
        x = batch  # sample from replay buffer
        result = self.forward(x) # mu, log_var, x, x_hat

        # loss function in model
        loss = self.model.loss_function(*result, kld_weight=self.kld_weight)

        self.log("TrainLoss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # if x , y(label) data
        # x, y = batch # sample from replay buffer
        # if x data (without label)
        x = batch  # sample from replay buffer
        result = self.forward(x)  # mu, log_var, x, x_hat

        # loss function in model
        loss = self.model.loss_function(*result, kld_weight=self.kld_weight)

        self.log("ValLoss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "ValLoss"}


class ConvVAE(nn.Module):
    def __init__(self, img_channels, latent_size):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size
        self.img_channels = img_channels

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=self.img_channels, out_channels=32, kernel_size=(4, 4), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2))
        # for encode Latent mean and std
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)
        self.fc_logsigma = nn.Linear(256 * 6 * 6, latent_size)

        # Decoder
        # The decoder will reconstruct the image to the size of the input image in order to calculate the loss function.
        # for decode reconstruction
        self.fc_recon = nn.Linear(latent_size, 256 * 6 * 6)
        self.deconv1 = nn.ConvTranspose2d(in_channels=256 * 6 * 6, out_channels=128, kernel_size=(5, 5), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 5), stride=(2, 2))
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(5, 5), stride=(2, 2))

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 256 * 6 * 6)

        mu = self.fc_mu(h)
        logsigma = self.fc_logsigma(h)

        return mu, logsigma # mean and logvar(->std)

    def decode(self, z):
        # h = self.fc3(z).view(-1, 256 * 6 * 6, 1, 1)
        h = F.relu(self.fc_recon(z))
        h = h.unsqueeze(-1).unsqueeze(-1)    # [batch, input, 1, 1]
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        reconstruction = F.sigmoid(self.deconv5(h))
        return reconstruction

    def reparameterize(self, mu, logsigma):
        # sample from mean and std
        std = torch.exp(0.5 * logsigma)     # std = sqrt(sigma)
        eps = torch.randn_like(std)         # random noise
        return eps * std + mu

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        recon_x = self.decode(z)
        return recon_x, mu, logsigma

    def loss_function(self, recon_x, x, mu, logsigma, kld_weight):
        batch_size = x.size(0)
        loss = F.binary_cross_entropy(recon_x, x, size_average=False)
        kld = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        loss /= batch_size
        kld /= batch_size
        return loss + kld_weight * kld.sum()

    def save_model(self, state, is_best, filename, best_filename):
        torch.save(state, filename)
        if is_best:
            torch.save(state, best_filename)





# def visualize_reconstructions(model, input_imgs):
#     model.eval()
#     with torch.no_grad():
#         mu, logvar, input_img, reconst_img = model(input_imgs.to(model.device))
#     input_img = input_img.view(1, 1, 28, 28)
#     reconst_img = reconst_img.view(1, 1, 28, 28)
#     reconst_img = reconst_img.cpu()
#
#     imgs = torch.stack([input_img, reconst_img], dim=1).flatten(0, 1)
#     grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
#     grid = grid.permute(1, 2, 0)
#     plt.figure(figsize=(7, 4.5))
#     plt.title(f"Reconstructed")
#     plt.imshow(grid)
#     plt.axis('off')
#     plt.show()
#
#
# if __name__ =="__main__":
#     global buffer_dict
#
#     # Setting the seed
#     pl.seed_everything(0)
#
#     # hyper parameter setting
#     hparameters = {"max_epochs": 50, "learning_rate": 0.003, "kld_weight": 0.00025, "batch_size": 64}
#
#     ###################################################################################
#     # MNIST Dataset # training_step과 validation_step에서 x, y = batch 를 사용해야 함.
#     # dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
#     # train, val = random_split(dataset, [55000, 5000])
#     # train_dataset = DataLoader(train, batch_size=64)
#     # val_dataset = DataLoader(val, batch_size=64)
#     #
#     # # model
#     # vae_model = VanillaVAE(input_dim=28*28, latent_dim=28)
#     ##################################################################################
#
#     # RVO simulation Dataset # training_step과 validation_step에서 x = batch 를 사용해야 함.
#     # If pretrain data exist, Get that.
#     PRETRAIN_BUFFER_PATH = '../vae_ckpts/buffer_dict.pkl'
#     if os.path.isfile(PRETRAIN_BUFFER_PATH):
#         print("Found Pretrain Data Buffer")
#         with open(PRETRAIN_BUFFER_PATH, 'rb') as f:
#             buffer_dict = pickle.load(f)
#
#     # replay buffer 의 상태 정보를 가져온다.
#     # tensor 작업
#     state_dataset = np.array(buffer_dict['vae'])
#     # 정규화 작업 (-1 ~ 1)
#     # state_dataset = normalize(state_dataset)
#     normalizer = Normalizer().fit(state_dataset)
#     state_dataset = normalizer.transform(state_dataset)
#     # Tensor
#     state_dataset = torch.from_numpy(state_dataset).float()
#     # train set, validation set
#     train_data_num = int(len(state_dataset) * 0.8)
#     val_data_num = len(state_dataset) - train_data_num
#     train, val = random_split(state_dataset, [train_data_num, val_data_num])
#
#     train_dataset = DataLoader(train,batch_size=hparameters['batch_size'], num_workers=5, drop_last=True)
#     val_dataset = DataLoader(val, batch_size=hparameters['batch_size'], num_workers=5, drop_last=True)
#
#     # model
#     vae_model = VanillaVAE(input_dim=32, latent_dim=8, hidden_dim=[32, 32, 32, 16, 16, 16, 8, 8])
#
#     ##################################################################################
#
#     # load check point
#     make_directories("vae_ckpts")
#     CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "vae_ckpts/")
#     ckpt_name = f'vae_RVO_epoch={hparameters["max_epochs"]}.pt'
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, ckpt_name)
#
#     if os.path.isfile(pretrained_filename):
#         print("Found Pretrained Model, loading...")
#         vae_model.load_state_dict(torch.load(pretrained_filename)["state_dict"])
#         # Set pytorch lightning
#         vae_exe_model = VAEEXE(vae_model=vae_model, **hparameters)
#     else:
#         # Set pytorch lightning
#         vae_exe_model = VAEEXE(vae_model=vae_model, **hparameters)
#         # Set trainer
#         trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0,
#                              auto_lr_find=True,
#                              max_epochs=hparameters["max_epochs"])
#         trainer.fit(vae_exe_model, train_dataset, val_dataset)
#         # save model
#         torch.save({"state_dict": vae_model.state_dict()}, pretrained_filename)

    # # MNIST Result Check
    # sample_img = torch.stack([dataset[15][0]], dim=0)
    # visualize_reconstructions(vae_exe_model, sample_img)
