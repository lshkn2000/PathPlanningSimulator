import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def make_file_list(folder_path):
    """
    :param folder_path:
    :return: file list with path
    """

    list_files = os.listdir(folder_path)
    file_list_with_path = []
    for file_name in range(list_files):
        file_with_path = folder_path + r'/' + file_name
        file_list_with_path.append(file_with_path)

    return file_list_with_path


class ImageTransform():
    """
    Image Tranform(tensor, resize, normalize) for training
    """
    def __init__(self, mean, std):
        """
        :param mean: [r, g, b]
        :param std: [r, g, b]
        """
        self.data_transform = transforms.Compose( # Compose 클래스를 통해 통합적으로 전처리
            [
                transforms.ToTensor(), # PIL, ndarray를 tensor로 변환
                transforms.Normalize(mean, std) # 이미지 정규화
            ]
        )

    def __call__(self, img):
        return self.data_transform(img)


class ImgDataset(Dataset):
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
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed


class ImgDatasetLabeled(Dataset):
    def __init__(self, x_file_list, y_file_list, transform):
        super(ImgDatasetLabeled, self).__init__()
        # 데이터 셋의 전처리 작업 진행
        self.x_data_file_list = x_file_list
        self.y_data_file_list = y_file_list
        self.transform = transform

    def __len__(self):
        # 데이터 셋의 길이 (총 샘플의 수)
        return len(self.file_list)

    def __getitem__(self, index):
        # 데이터 셋의 특정 1개의 샘플 가져오기
        img_path = self.x_data_file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = torch.FloatTensor(self.y_data_file_list[index])

        return img_transformed, label


# DataLoader 에 커스텀한 데이터 셋 넣기
train_img_list = make_file_list(folder_path="file_path") # 학습 시킬 데이터 파일들 경로 가져오기

# 이미지 정규화
mean = (0.5, )
std = (0.3, )

train_dataset = ImgDataset(file_list=train_img_list,
                           transform=ImageTransform(mean, std))

train_dataloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True)

# iter와 next를 통해 순차적으로 뽑아올 수 있다.
batch_iterator = iter(train_dataloader)
images = next(batch_iterator)
#
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

