from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pytorch_lightning as pl

"""
################ pytorch lightning 을 이용하지 않으면 ################

torch_model = PytorchModel()

output = torch_model(x)
loss = loss_function(output, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

################ pytorch lightning 을 사용하면 ################

torch_model = PytorchModel()
trainer = pl.Trainer()
trainer.fit(torch_model, train_data, val_data)

"""


class CNNModel(pl.LightningModule):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnv = None
        self.fn = None

        self.act_relu = nn.ReLU()
        self.act_lklrelu = nn.LeakyReLU()
        self.bn = None
        self.maxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.softmax = nn.Softmax()

    def model_setting(self,
                            img_size: tuple,
                            hidden_dims: List,
                            output_dim: int,
                            in_channels: int,
                            out_channels: int,
                            kernel_size: tuple = (3, 3),
                            stride: tuple = (1, 1),
                            padding: tuple = (0, 0),
                            dilation: tuple = (1, 1),
                            ):
        """
        Conv2d의 이미지 입력은 [커널 갯수 = 출력채널, 입력 채널, height, width]

        :param img_size: (height, width)
        :param hidden_dims:
        :param output_dim:

        :param in_channels:
        :param out_channels:

        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :return:

        """
        self.cnv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=(1, 1),
                             padding=(0, 0),
                             dilation=(1, 1),
                             bias=True,
                             )

        cnv_height = ((img_size[0] + 2 * padding[0] - dilation[0] * kernel_size[0]) / stride[0]) + 1
        cnv_width = ((img_size[1] + 2 * padding[1] - dilation[1] * kernel_size[1]) / stride[1]) + 1

        fn_layer = [int(out_channels * cnv_height * cnv_width)] + hidden_dims + [output_dim]

        fn = []
        for i in range(len(fn_layer) - 1):
            fn.append(
                nn.Sequential(
                    nn.Linear(fn_layer[i], fn_layer[i + 1], bias=True),
                    # nn.BatchNorm1d(fn_layer[i + 1], momentum=0.7),    # batch size 가 2 이상이어야 정규화 크기를 정할 수 있다.
                    nn.LeakyReLU(),
                )
            )
        self.fn = nn.Sequential(*fn)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act_lklrelu(self.bn(self.cnv(x)))
        x = self.flat(x)
        out = self.fn(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_function(self, out, target):
        return F.nll_loss(out, target)

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        out = self.forward(x)
        loss = self.loss_function(out, y)
        return loss
