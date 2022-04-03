import os
import random

import torch
import torchvision.utils
from torch import nn, optim
from torch.nn import functional as F
from typing import List
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class VanilaVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super(VanilaVAE, self).__init__()

        self.kld_weight = 0.00025
        self.encoder = nn.Sequential(
                                    nn.Linear(28*28, 196),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(196, momentum=0.7),
                                    nn.Linear(196, 49),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(49, momentum=0.7),
                                    nn.Linear(49, 28),
                                    nn.LeakyReLU()
                                )
        self.latent2mu = nn.Linear(28, 28)
        self.latent2log_var = nn.Linear(28, 28)
        self.decoder = nn.Sequential(
            nn.Linear(28, 49),
            nn.ReLU(),
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 784),
            nn.Tanh()
        )

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
        x = x.view(x.size(0), -1)
        mu, log_var = self.encode(x)
        latent = self.reparameterize(mu, log_var)
        output = self.decoder(latent)
        return mu, log_var, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, log_var, x_hat = self.forward(x)

        recons_loss = F.mse_loss(x, x_hat)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, log_var, x_hat = self.forward(x)

        recons_loss = F.mse_loss(x, x_hat)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss

        self.log('val_kld_loss', kld_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recons_loss, on_step=False, on_epoch=True)
        self.log('vla_loss', loss, on_step=False, on_epoch=True)
        return x_hat, loss

    def validation_epoch_end(self, outputs):
        if not os.path.exists('test_img'):
            os.makedirs('test_img')
        choice = random.choice(outputs) # choose random batch from outputs
        output_sample = choice[0]
        output_sample = output_sample.reshape(-1, 1, 28, 28)
        save_image(output_sample, f'test_img/test.png')

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, log_var, x_hat = self.forward(x)

        recons_loss = F.mse_loss(x, x_hat)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return x_hat, loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)

        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def compare_imgs(img1, img2, title_prefix=""):
    loss = F.mse_loss(img1, img2, reduction='sum')

    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, value_range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss : {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def visualize_reconstructions(model, input_imgs):
    model.eval()
    with torch.no_grad():
        mu, logvar, reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.view(1, 1, 28, 28)
    reconst_imgs = reconst_imgs.cpu()

    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title(f"Reconstructed")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def get_train_images(num):
    return torch.stack([dataset[num][0]], dim=0)


def train_vae(latent_dim, ckpt_name: str, max_epochs=100):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, ckpt_name),
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_train_images(8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True     # log tensorboard

    # get pretrained model
    pretrained_filename = os.path.join(CHECKPOINT_PATH, ckpt_name)
    if os.path.isfile(pretrained_filename):
        print("Found Pretrained Model, loading...")
        model = VanilaVAE.load_from_checkpoint(pretrained_filename)
    else:
        model = VanilaVAE(**hyper_parameters)
        trainer.fit(model, DataLoader(train, batch_size=64), DataLoader(val, batch_size=64))

    val_result = trainer.test(model, test_dataloaders=DataLoader(val, batch_size=64), verbose=False)
    result = {"val": val_result}
    return model, result

if __name__ == '__main__':
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "lightning_logs/version_23/checkpoints/epoch=49-step=42999.ckpt")

    # Setting the seed
    pl.seed_everything(0)

    # # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    hyper_parameters = {'kld_weight': 0.00025}

    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    vae_lightning = VanilaVAE(**hyper_parameters)
    # trainer = pl.Trainer(gpus=1, auto_lr_find=True, max_epochs=50)
    # trainer.fit(vae_lightning, DataLoader(train, batch_size=64), DataLoader(val, batch_size=64))

    model = vae_lightning.load_from_checkpoint(CHECKPOINT_PATH)
    # val_result = trainer.test(model, DataLoader(train, batch_size=64), verbose=False)

    input_imgs = get_train_images(10)
    visualize_reconstructions(model, input_imgs)

    # test1
    # img_x, label_y = dataset[0]
    # compare_imgs(img_x, img_x, 'test')



