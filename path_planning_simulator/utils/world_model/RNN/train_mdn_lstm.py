import os

from torch.utils.data import BatchSampler, SequentialSampler
from tqdm import tqdm
import cv2
import torch
import torchvision.transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import gc
gc.collect()
torch.cuda.empty_cache()

from path_planning_simulator.utils.world_model.VAE.make_dataset import *
from path_planning_simulator.utils.world_model.VAE import VAE
from path_planning_simulator.utils.world_model.RNN.mdn_lstm import MDNLSTM


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

z_size = 32  # (== vae z_dim !)
n_lstm_hidden = 256
n_gaussians = 64    # few gaussian (hard to predict) <-> many gaussian (hard to interpret)

batch_size = 61  # resize according to input sizes
seq_len = 16    # lstm sequence length
epochs = 500

PATH = r'/home/rvlab/PathPlanningSimulator_branch/PathPlanningSimulator_new_worldcoord_2/path_planning_simulator'

def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
############################## 학습된 VAE 가져오기 ####################################
# # 이미지 정보 가져오기
# PATH = r'/home/rvlab/PathPlanningSimulator_branch/PathPlanningSimulator_new_worldcoord_2/path_planning_simulator'
# PRETRAIN_IMG_PATH = os.path.join(PATH, r'vae_ckpts/img_dataset')
# train_img_list = make_file_list(folder_path=PRETRAIN_IMG_PATH)  # 학습 시킬 데이터 파일들 경로 가져오기
#
# # RVO2 dataset 원본 가져오기
# PRETRAIN_BUFFER_PATH = os.path.join(PATH, 'vae_ckpts/simulation_buffer_dict.pkl')
# if os.path.isfile(PRETRAIN_BUFFER_PATH):
#     with open(PRETRAIN_BUFFER_PATH, 'rb') as f:
#         buffer_dict = pickle.load(f)
#         train_offline_trajectories_list = buffer_dict["pretrain"]
# else:
#     assert Exception("RVO2를 통해 얻은 Offline Raw Dataset 이 없습니다.")
#
#
# VAE 모델 가져오기
vae_model = VAE.ConvVAE(img_channels=3, latent_dim=z_size).to(device=device)
VAE_MODEL_PATH = os.path.join(PATH, r'utils/world_model/VAE/vae_models/best.tar')
# VAE_MODEL_PATH = os.path.join(PATH, r'utils/world_model/VAE/vae_models/monkey5_best.tar')
if os.path.exists(VAE_MODEL_PATH):
    print("Load VAE Weights")
    state = torch.load(VAE_MODEL_PATH)
    vae_model.load_state_dict(state['state_dict'])
vae_model.to(device)
vae_model.eval()


###############################MDN-LSTM 모델 만들기 ##################################
mdn_lstm_model = MDNLSTM(z_size, n_lstm_hidden, n_gaussians)
mdn_lstm_model = mdn_lstm_model.to(device)
optimizer = torch.optim.Adam(mdn_lstm_model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

# make model save folder
mdn_lstm_model_dir = os.path.join(PATH, r'utils/world_model/RNN/mdn_lstm_model')
make_directories(mdn_lstm_model_dir)

# load model
load_mdn_lstm_model = os.path.join(mdn_lstm_model_dir, 'best.tar')
# load_mdn_lstm_model = os.path.join(mdn_lstm_model_dir, 'monkey5_best.tar')
if os.path.exists(load_mdn_lstm_model):
    print("Loading MDN LSTM Weights")
    state = torch.load(load_mdn_lstm_model)
    mdn_lstm_model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

###############################GET Latent Dataset #################################
print("Loading Latent Dataset ...")
LATENT_DATASET_PATH = os.path.join(PATH, r'utils/world_model/VAE/vae_models/latent_dataset.pkl')
# LATENT_DATASET_PATH = os.path.join(PATH, r'utils/world_model/VAE/vae_models/monkey5_latent_dataset.pkl')
if os.path.isfile(LATENT_DATASET_PATH):
    with open(LATENT_DATASET_PATH, 'rb') as f:
        latent_dict = pickle.load(f)


z = np.array(latent_dict['z'])
z = torch.from_numpy(z).to(torch.float32).squeeze()
z = z.view(batch_size, -1, z.size(1))   # z.size(1) == z_dim

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


def save_model(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def train_mdn_lstm():
    for epoch in range(epochs):
        total_loss = 0
        hidden = mdn_lstm_model.init_hidden(batch_size)
        for i in range(0, z.size(1) - seq_len, seq_len):
            inputs = z[:, i:i+seq_len, :].to(device)
            targets = z[:, (i+1):(i+1)+seq_len, :].to(device)

            hidden = detach(hidden)
            (pi, mu, sigma), hidden = mdn_lstm_model(inputs, hidden)
            loss = mdn_lstm_model.mdn_loss_function(pi, sigma, mu, targets)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

        scheduler.step(total_loss / z.size(0))

        # check point
        file_name = os.path.join(mdn_lstm_model_dir, 'best.tar')
        # file_name = os.path.join(mdn_lstm_model_dir, 'monkey5_best.tar')
        save_state = {
            'epoch': epoch,
            'state_dict': mdn_lstm_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(save_state, file_name)

    # load model
    load_mdn_lstm_model = os.path.join(mdn_lstm_model_dir, 'best.tar')
    # load_mdn_lstm_model = os.path.join(mdn_lstm_model_dir, 'monkey5_best.tar')
    if os.path.exists(load_mdn_lstm_model):
        print("Loading MDN LSTM Weights")
        state = torch.load(load_mdn_lstm_model)
        mdn_lstm_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])


################# TRAIN #################
train_mdn_lstm()

########## TEST SAMPLING IMG ############
# make predicted test img save folder
mdn_lstm_predicted_test_img_dir = os.path.join(PATH, r'utils/world_model/RNN/mdn_lstm_model/test_samples')
# mdn_lstm_predicted_test_img_dir = os.path.join(PATH, r'utils/world_model/RNN/mdn_lstm_model/monkey5_test_samples')
make_directories(mdn_lstm_predicted_test_img_dir)

# sequence img shot
# batch_length = 1
batch_length = z.size(0)
for batch_idx in range(batch_length):
    seq_idx = z.size(1)
    for seq_idx in tqdm(range(seq_idx - 1)):
        input = z[batch_idx:batch_idx + 1, seq_idx:seq_idx + 1, :].to(device)
        target = z[batch_idx:batch_idx + 1, seq_idx + 1:seq_idx + 2, :].to(device)

        hidden = mdn_lstm_model.init_hidden(1)
        (pi, mu, sigma), _ = mdn_lstm_model(input, hidden)
        target_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(n_gaussians)]

        sample_input = vae_model.decode(input)
        sample_target = vae_model.decode(target)
        sample_target_predict = vae_model.decode(torch.cat(target_preds))
        z_compare_data = torch.cat([input, target] + target_preds)
        compare_x = vae_model.decode(z_compare_data)

        save_image(compare_x.data.cpu(), mdn_lstm_predicted_test_img_dir + r'/sample_image_' + 'batch_' + str(batch_idx) + '_seq_'+ str(seq_idx) + '.png')


# only oneshot
# batch_idx = np.random.randint(z.size(0))
# seq_idx = np.random.randint(z.size(1))
# input = z[batch_idx:batch_idx+1, seq_idx:seq_idx+1, :].to(device)
# target = z[batch_idx:batch_idx+1, seq_idx+1:seq_idx+2, :].to(device)
#
# hidden = mdn_lstm_model.init_hidden(1)
# (pi, mu, sigma), _ = mdn_lstm_model(input, hidden)
#
# target_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(n_gaussians)]
#
# print(target_preds[0].shape)
# sample_input = vae_model.decode(input)
# sample_target = vae_model.decode(target)
# sample_target_predict = vae_model.decode(torch.cat(target_preds))
#
# z_compare_data = torch.cat([input, target] + target_preds)
# compare_x = vae_model.decode(z_compare_data)
# print(compare_x.shape)
#
# save_image(sample_input.data.cpu(), 'sample_input.png')
# save_image(sample_target.data.cpu(), 'sample_target.png')
# save_image(sample_target_predict.data.cpu(), 'sample_target_predict.png')
# save_image(compare_x.data.cpu(), 'sample_image1.png')
# sample_input_img = Image.open('sample_input.png')
# sample_target_img = Image.open('sample_target.png')
# sample_target_predict_img = Image.open('sample_target_predict.png')
#
# sample_input_img.show()
# sample_target_img.show()
# sample_target_predict_img.show()
