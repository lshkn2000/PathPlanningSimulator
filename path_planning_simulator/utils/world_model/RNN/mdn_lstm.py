import torch
import torch.nn as nn
import torch.nn.functional as F
from path_planning_simulator.utils.world_model.const import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MDN_LSTM(nn.Module):
    def __init__(self, sequence_len, lstm_hidden_dim, z_dim, num_layers, n_gaussians, fc_hidden_dim):
        super(MDN_LSTM, self).__init__()

        self.n_gaussians = n_gaussians
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.sequence_len = sequence_len

        # init lstm hidden
        self.hidden = self.init_hidden(self.sequence_len)

        # Encoding
        self.fc1 = nn.Linear(self.z_dim + 1, self.fc_hidden_dim)
        self.lstm = nn.LSTM(self.fc_hidden_dim, lstm_hidden_dim, num_layers)    # input_dim, hidden_dim, num_layer

        # MDN Output
        self.z_pi = nn.Linear(lstm_hidden_dim, n_gaussians * self.z_dim)
        self.z_sigma = nn.Linear(lstm_hidden_dim, n_gaussians * self.z_dim)
        self.z_mu = nn.Linear(lstm_hidden_dim, n_gaussians * self.z_dim)

    def init_hidden(self, sequence):
        hidden = torch.zeros(self.num_layers, sequence, self.lstm_hidden_dim, device=device)
        cell = torch.zeros(self.num_layers, sequence, self.lstm_hidden_dim, device=device)
        return hidden, cell

    def forward(self, x):
        self.lstm.flatten_parameters()

        x = F.relu(self.fc1(x))
        z, self.hidden = self.lstm(x, self.hidden)
        sequence = x.size()[1]

        # (batch_size, sequence_length, n_gaussians, latent_dimension)
        pi = self.z_pi(z).view(-1, sequence, self.n_gaussians, self.z_dim)
        pi = F.softmax(pi, dim=2)
        sigma = torch.exp(self.z_sigma(z)).view(-1, sequence, self.n_gaussians, self.z_dim)
        mu = self.z_mu(z).view(-1, sequence, self.n_gaussians, self.z_dim)
        return pi, sigma, mu

    def gaussian_distribution(self, y, mu, sigma):
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = torch.sum(result, dim=1)
        result = -torch.log(result)
        return torch.mean(result)

    def mdn_loss_function(self, pi, sigma, mu, y):
        result = self.gaussian_distribution(y, mu, sigma) * pi
        result = torch.sum(result, dim=1)
        result = -torch.log(result)
        return torch.mean(result)

    # def mdn_loss_function(self, out_pi, out_sigma, out_mu, y):
    #     y = y.view(-1, SEQUENCE, 1, LATENT_VEC)
    #     result = Normal(loc=out_mu, scale=out_sigma)
    #     result = torch.exp(result.log_prob(y))
    #     result = torch.sum(result * out_pi, dim=2)
    #     result = -torch.log(EPSILON + result)
    #     return torch.mean(result)


