import torch
import torch.nn as nn
import torch.nn.functional as F
from path_planning_simulator.utils.world_model.const import *


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class MDNLSTM(nn.Module):
    def __init__(self, z_size, sequence_length, n_lstm_hidden, n_gaussians=5, n_layers=1):
        super(MDNLSTM, self).__init__()

        self.z_size = z_size        # input size
        self.n_lstm_hidden = n_lstm_hidden
        self.n_gaussians = n_gaussians
        self.n_layer = n_layers
        self.sequence_length = sequence_length

        # Encoding
        self.lstm = nn.LSTM(z_size, n_lstm_hidden, n_layers, batch_first=True)    # input_dim, hidden_dim, num_layer
        # batch first이므로 LSTM모듈의 입력은 [batch size, sequence length, input size] 형태이어야 한다.
        # out : [batch size, sequence length, hidden size]
        # 입력을 시퀀스 처리해야한다. ex) 28 X 28 이미지이면 sequence : 28, input size : 28

        # MDN Output
        self.z_pi = nn.Linear(n_lstm_hidden, n_gaussians * z_size)
        self.z_mu = nn.Linear(n_lstm_hidden, n_gaussians * z_size)
        self.z_logsigma = nn.Linear(n_lstm_hidden, n_gaussians * z_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layer, batch_size, self.n_lstm_hidden, device=device)
        cell = torch.zeros(self.n_layer, batch_size, self.n_lstm_hidden, device=device)
        return hidden, cell

    def get_mixture_distribution(self, y):
        rollout_length = y.size(1)      # y : [batch_size, sequence_length, hidden_size]
        pi = self.z_pi(y)
        mu = self.z_mu(y)
        logsigma = self.z_logsigma(y)

        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        logsigma = logsigma.view(-1, rollout_length, self.n_gaussians, self.z_size)

        pi = F.softmax(pi, 2)
        sigma = torch.exp(logsigma)
        return pi, mu, sigma

    def forward(self, x, hidden):
        # input : [batch_size, sequence_length, input_size]
        self.lstm.flatten_parameters()

        pred_z, (h, c) = self.lstm(x, hidden)     # pred_z (=output) : [batch_size, sequence_length, hidden_size]
        pi, mu, sigma = self.get_mixture_distribution(pred_z)
        return (pi, mu, sigma), (h, c)     # (hidden, cell)

    def mdn_loss_function(self, pi, mu, sigma, y):
        # pi, mu, sigma : predict / y : target
        y = y.unsqueeze(2)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss)
        return loss.mean()


