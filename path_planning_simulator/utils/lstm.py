import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class LSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(LSTM, self).__init__()
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim

		self.init_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True).to(device)

	def forward(self, x):
		# input X : (batch_size, seq_len, input_dim) -> (batch_size, ped_num, ped_data)
		batch_size = x.size(0)
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
		cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

		output, (hn, cn) = self.init_lstm(x, (hidden, cell))
		output = hn[0]


		return output

# model = LSTM(input_dim=5, output_dim= 20, hidden_dim=10, num_layers=2)
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # input_data = input_data.reshape(-1, sequence_length, input_dime)
# outputs = model(input_data)