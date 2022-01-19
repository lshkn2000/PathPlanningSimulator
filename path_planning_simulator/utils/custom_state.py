import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from path_planning_simulator.utils.lstm import LSTM

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class FeaturedLSTM:
    def __init__(self, input_dim=5, output_dim=10+7, hidden_dim=30, num_layers=1):
        self.lstm = LSTM(input_dim=input_dim, output_dim=output_dim - 7, hidden_dim=hidden_dim - 7, num_layers=num_layers)

        self.fc_1 = nn.Linear(hidden_dim, 128).to(device)
        self.fc_2 = nn.Linear(128, output_dim).to(device)

    def custom_state_for_lstm(self, state):
        if state.ndim == 1: # numpy : 시뮬레이션에서 numpy로 받아
            robot_state = state[:7]  # px, py, vx, vy, gx, gy, r
            obstacles_states = state[7:]  # (px, py, vx, vy, r) * ped_num
            obstacles_num = len(obstacles_states) // 5
            obstacles_states = obstacles_states.reshape(-1, obstacles_num, 5)
            robot_state_tensor = torch.tensor(robot_state, device=device, dtype=torch.float32)
            obstacles_states = torch.tensor(obstacles_states, device=device, dtype=torch.float32)
            outputs = self.lstm(obstacles_states)
            outputs = outputs.squeeze()
            flatten_output = torch.cat([robot_state_tensor, outputs], dim=-1).unsqueeze(0)

            flatten_output = F.relu(self.fc_1(flatten_output))
            flatten_output = self.fc_2(flatten_output)

            return flatten_output

        elif state.dim() == 2: # tensor : 리플레이 버퍼에서 텐서처리 후 가져옴
            batch_size = state.shape[0]

            # 동적 장애물만 있다는 가정에서 만든 것.
            # 정적 장애물 정보처리는 나중에 추가예정
            robot_state = state[:, :7]         # px, py, vx, vy, gx, gy, r
            obstacles_states = state[:, 7:]    # (px, py, vx, vy, r) * ped_num
            obstacles_num = obstacles_states.shape[1] // 5
            obstacles_states = obstacles_states.reshape(batch_size, obstacles_num, 5).to(device)
            # robot_state_tensor = torch.tensor(robot_state, dtype=torch.float32)
            # obstacles_states = torch.tensor(obstacles_states, device=device, dtype=torch.float32)
            # lstm input : (batch_size, seq_len, input_size)

            # learning_rate = 0.001
            # lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
            outputs = self.lstm(obstacles_states)       # output : (batch_size, seq_length, hidden_layer)
            # outputs = outputs.squeeze()

            flatten_output = torch.cat([robot_state, outputs], dim=-1)

            flatten_output = F.relu(self.fc_1(flatten_output))
            flatten_output = self.fc_2(flatten_output)

            return flatten_output