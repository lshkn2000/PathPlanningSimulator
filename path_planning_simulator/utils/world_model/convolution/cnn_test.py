import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils

from PIL import Image

import matplotlib.pyplot as plt


class CNNTest(nn.Module):
    def __init__(self, input_channels):
        super(CNNTest, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

    def forward(self, x):
        print(f" img size : {x.size()}")
        x = F.relu(self.conv1(x))
        print(f"conv1 size : {x.size()}")
        x = F.relu(self.conv2(x))
        print(f"conv2 size : {x.size()}")
        x = F.relu(self.conv3(x))
        print(f"conv3 size : {x.size()}")
        x = F.relu(self.conv4(x))
        print(f"conv4 size : {x.size()}")
        x = x.view(x.size(0), -1)
        print(f"flatten size : {x.size()}")

        return x


path = "/home/huni/PathPlanningSimulator_new_worldcoord/path_planning_simulator/utils/vae_images/test.png"
img = Image.open(path)

# PIL로부터 텐서로 변환
tf_to_tensor = transforms.ToTensor()
img_tensor = tf_to_tensor(img)  # torch.Size([3, 242, 242])
img_tensor = img_tensor.unsqueeze(0) # img size : torch.Size([1, 3, 242, 242])

cnn_test = CNNTest(3)
conved_img = cnn_test(img_tensor)