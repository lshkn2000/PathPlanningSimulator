import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import pickle

"""
World Model 학습
실행 순서는 make_img -> make_img_dataset -> cnn_vae_train
학습에 사용할 데이터 셋과 데이터 로더를 만들기 위한 함수들이다. 
아래의 사용 예시를 참고하여라
"""



def make_file_list(folder_path):
    """
    :param folder_path:
    :return: file list with path
    """

    list_files = os.listdir(folder_path)
    file_list_with_path = []
    for file_name in list_files:
        file_with_path = folder_path + r'/' + file_name
        file_list_with_path.append(file_with_path)

    file_list_with_path.sort()
    return file_list_with_path


class ImageTransform():
    """
    Image Tranform(tensor, resize, normalize) for training
    """
    def __init__(self):
        """
        :param mean: [r, g, b]
        :param std: [r, g, b]
        """

        self.data_transform = transforms.Compose( # Compose 클래스를 통해 통합적으로 전처리
            [
                transforms.ToTensor(),  # PIL, ndarray(cv2)를 tensor로 변환
                transforms.Resize((64, 64)),

            ]
        )

    def __call__(self, img):
        return self.data_transform(img)


class ImgDataset(Dataset):
    """
    VAE 용 Img dataset 구축
    """
    def __init__(self, file_list, transform):
        super(ImgDataset, self).__init__()
        # 데이터 셋의 전처리 작업 진행
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        # 데이터 셋의 길이 (총 샘플의 수)
        return len(self.file_list)

    def __getitem__(self, index):
        # 데이터 셋의 특정 1개의 샘플 가져오기
        img_path = self.file_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_transformed = self.transform(img)
        return img_transformed


class TrajectoryDataset(Dataset):
    """
    Offline Trajectories
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        trajectory = self.file_list[index]
        state = trajectory[0]
        action = trajectory[1]
        next_state = trajectory[3]
        return (state, action, next_state)


################################### 사용 예시  ####################################
# DataSet을 만들 데이터 경로 가져오기
# PATH = r'/home/huni/PathPlanningSimulator_new_worldcoord/path_planning_simulator'
# PRETRAIN_IMG_PATH = os.path.join(PATH, r'vae_ckpts/img_dataset')
# train_img_list = make_file_list(folder_path=PRETRAIN_IMG_PATH) # 학습 시킬 데이터 파일들 경로 가져오기
#
#
# # 커스텀 데이터 셋에 데이터 넣어주기
# # 이미지 정규화
# mean = (0.5, )
# std = (0.3, )
#
# train_dataset = ImgDataset(file_list=train_img_list,
#                            transform=ImageTransform())
#
# train_dataloader = DataLoader(train_dataset,
#                               batch_size=64,
#                               shuffle=True)

# CHECK
# iter와 next를 통해 순차적으로 뽑아올 수 있다.
# batch_iterator = iter(train_dataloader)
# images = next(batch_iterator)
# grid = torchvision.utils.make_grid(images)
# plt.imshow(grid.permute(1,2,0))
# plt.show()
#
# LEARNNG EXAMPLE
# n_epochs = 20
# for epoch in range(n_epochs + 1):
#     for batch_idx, samples in enumerate(train_dataloader):
#         x_train, y_train = samples
#
#         # H(x) 계산
#         prediction = model(x_train) # 신경망 모델
#
#         # loss = cost 계산
#         loss = loss_function(prediction, y_train) # e.g. F.mse_loss()
#
#         # loss = cost 로 H(x) 계산 (backpropagation)
#         optimizer.zero_grad() # e.g. Adam optimizer
#         loss.backward()
#         optimizer.step()

