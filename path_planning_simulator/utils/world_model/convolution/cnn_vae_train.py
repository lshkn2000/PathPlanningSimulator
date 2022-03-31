import os

import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms

from path_planning_simulator.utils.world_model.convolution import VAE


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# argument
n_epochs = 20
vae_model_dir = os.getcwd() + r'/vae_models'

# dataset_train

# dataset_test

# data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

vae_model = VAE.ConvVAE(img_channels=3, latent_size=5).to(device=device)
optimizer = optim.Adam(vae_model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)


def train(epoch):
    vae_model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_dataloader):
        x = data.to(device)
        # predict
        recon_x, mu, logsigma = vae_model(data)
        # cost
        loss = vae_model.loss_function(recon_x, x, mu, logsigma, 0.1)
        # validate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_dataloader.dataset)))


def test():
    vae_model.train()
    test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            x = data.to(device)
            recon_x, mu, logsigma = vae_model(x)
            test_loss += vae_model.loss_function(recon_x, x, mu, logsigma, 0.1).item()

    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# VAE 모델 저장 폴더 확인 및 기존 모델 가져오기
if not os.path.exists(vae_model_dir):
    os.mkdir(vae_model_dir)

load_vae_model = os.path.join(vae_model_dir, 'best.tar')
if os.path.exists(load_vae_model):
    state = torch.load(load_vae_model)
    vae_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])


#########TRAIN###########
cur_best_model = None
is_best = False

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)

    # check point
    best_filename = os.path.join(vae_model_dir, 'best.tar')
    file_name = os.path.join(vae_model_dir, 'checkpoint.tar')
    is_best = not cur_best_model or test_loss < cur_best_model
    if is_best:
        cur_best_model = test_loss

    vae_model.model_save({
        'epoch': epoch,
        'state_dict': vae_model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, file_name, is_best, best_filename)



