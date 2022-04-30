import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


class TEST_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(TEST_LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        pred, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc(pred[:, -1, :])
        return out

    def init_hidden(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0, c0

    def loss_function(self, y_pred, y_target):
        loss = self.loss_fn(y_pred, y_target)
        return loss


samsung = fdr.DataReader('005930')
stock = samsung

print(stock.tail())

plt.figure(figsize=(16, 9))
sns.lineplot(y=stock['Close'], x=stock.index)
plt.xlabel('time')
plt.ylabel('price')
plt.show()

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled = scaler.fit_transform(stock[scale_cols])

df = pd.DataFrame(scaled, columns=scale_cols)
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', axis=1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
print(f"x train shape : {x_train.shape} y train shape : {y_train.shape}")
print(f"x train data : \n {x_train.head()}")


def make_seq_dataset(x_data, y_data, seq_len=20):
    x_list = []
    y_list = []
    for idx in range(len(x_data) - seq_len):
        x_list.append(np.array(x_data.iloc[idx:idx+seq_len]))
        y_list.append(np.array(y_data.iloc[idx+seq_len]))

    return np.array(x_list), np.array(y_list)


data_X, data_Y = make_seq_dataset(x_train, y_train)
train_data, train_label = data_X[:-300], data_Y[:-300]
test_data, test_label = data_X[-300:], data_Y[-300:]
X_train = torch.from_numpy(train_data).float()
Y_train = torch.from_numpy(train_label).float().unsqueeze(1)

lstm_model = TEST_LSTM(4, 30, 1, num_layers=2)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

epochs = 200
batch_size = 64
loss_hist = np.zeros(epochs)
for epoch in range(epochs):

    y_train_pred = lstm_model(X_train)

    loss = lstm_model.loss_fn(y_train_pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 and epoch != 0:
        print("Epoch ", epoch, "MSE: ", loss.item())
    loss_hist[epoch] = loss.item()

## Train Fitting
plt.plot(y_train_pred.detach().numpy(), label="Preds")
plt.plot(Y_train.detach().numpy(), label="Real")

plt.legend()
plt.show()