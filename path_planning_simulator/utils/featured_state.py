import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, ped_state):
        featured_state = F.relu(self.fc1(ped_state))
        featured_state = self.fc2(featured_state)
        return featured_state


class FeaturedState(nn.Module):
    def __init__(self, batch_size:int):
        super(FeaturedState, self).__init__()

        self.batch_size = batch_size

        # featured observation
        self.obstacle_feature = MLP(5, 3)
        self.robot_feature = MLP(7, 4)

        # featured weight
        self.lstm = nn.LSTM(3, 1, batch_first=True)

        # featured softmax
        self.softmax = nn.Softmax(dim=1)

    def forward_(self, state):
        if isinstance(state, np.ndarray):
            robot_state = state[:7]
            obstacles_states = state[7:]  # (px, py, vx, vy, r) * ped_num
            obstacles_num = len(obstacles_states) // 5
            obstacles_states = obstacles_states.reshape(-1, obstacles_num, 5)  # (batch size, ped num, ped info)
            robot_state = torch.tensor(robot_state, device=device, dtype=torch.float32)
            obstacles_states = torch.tensor(obstacles_states, device=device, dtype=torch.float32)
            robot_state = self.robot_feature(robot_state)
            obstacles_states = self.obstacle_feature(obstacles_states)
            return robot_state, obstacles_states
        elif isinstance(state, torch.Tensor):
            batch_size = state.shape[0]
            robot_states = state[:, :7]  # px, py, vx, vy, gx, gy, r
            obstacles_states = state[:, 7:]  # (px, py, vx, vy, r) * ped_num
            obstacles_num = obstacles_states.shape[1] // 5
            robot_states = robot_states.reshape(batch_size, -1, 7).to(device)
            obstacles_states = obstacles_states.reshape(batch_size, obstacles_num, 5).to(device)
            robot_states = self.robot_feature(robot_states)
            obstacles_states = self.obstacle_feature(obstacles_states)
            return robot_states, obstacles_states

    def forward(self, state):
        robot_featured_state, ob_featured_state = self.forward_(state)
        ob_lstm_featured_state, hidden = self.lstm(ob_featured_state)
        weights = self.softmax(ob_lstm_featured_state)
        ob_weighted_featured_state = weights * ob_featured_state
        if ob_weighted_featured_state.shape[0] > 1:
            ob_flatten_weighted_featred_state = ob_weighted_featured_state.reshape(self.batch_size, 1, -1)
            output = torch.cat((robot_featured_state, ob_flatten_weighted_featred_state), dim=-1).squeeze()
            return output
        else:
            robot_flatten_featured_state = robot_featured_state.reshape(1,-1)
            ob_flatten_weighted_featred_state = ob_weighted_featured_state.reshape(1,-1)
            output = torch.cat((robot_flatten_featured_state, ob_flatten_weighted_featred_state), dim=-1)
            return output








