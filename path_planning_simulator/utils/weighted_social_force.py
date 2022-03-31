import torch
import torch.nn as nn
import torch.nn.functional as F

from path_planning_simulator.utils.lstm import LSTM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 0. 각 보행자들의 FCL 을 통해 특징 추출
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, ped_state):
        featured_state = F.relu(self.fc1(ped_state))
        featured_state = self.fc2(featured_state)
        return featured_state

# 1. 각 보행자들의 정보를 LSTM을 통해 주변인들과의 관계 Feature 추출 (Scalar)
class PedsFeature(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # (보행자 정보(5), hidden_dim, 특징벡터)
        super(PedsFeature, self).__init__()
        self.ped_feature = MLP(input_dim, output_dim)
        self.lstm = LSTM(input_dim=output_dim, hidden_dim=hidden_dim, num_layers=1)
        self.ped_weighted = MLP(hidden_dim + output_dim, 1)     # (LSTM 통해 보행자들 사이 특징 벡터 + 각 보행자 특징벡터 dim, weight(scalar))

        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=3e-4)
        self.ped_lstm_optimizer = torch.optim.Adam(self.ped_lstm.parameters(), lr=3e-4)
        self.ped_weighted_optimizer = torch.optim.Adam(self.ped_weighted.parameters(), lr=3e-4)

    def forward_(self, state):
        if state.ndim == 1:  # numpy : 시뮬레이션에서 numpy로 받아
            robot_state = state[:7]  # px, py, vx, vy, gx, gy, r
            obstacles_states = state[7:]  # (px, py, vx, vy, r) * ped_num
            obstacles_num = len(obstacles_states) // 5
            obstacles_states = obstacles_states.reshape(-1, obstacles_num, 5) # (batch size, ped num, ped info)
            obstacles_states = torch.tensor(obstacles_states, device=device, dtype=torch.float32)
            obstacles_states = self.ped_feature(obstacles_states)
            return obstacles_states

        elif state.dim() == 2: # tensor : 리플레이 버퍼에서 텐서처리 후 가져옴
            batch_size = state.shape[0]
            robot_state = state[:, :7]  # px, py, vx, vy, gx, gy, r
            obstacles_states = state[:, 7:]  # (px, py, vx, vy, r) * ped_num
            obstacles_num = obstacles_states.shape[1] // 5
            obstacles_states = obstacles_states.reshape(batch_size, obstacles_num, 5).to(device)
            obstacles_states = self.ped_feature(obstacles_states)
            return obstacles_states

    def forward(self, state):
        featured_peds = self.forward_(state)           # output : (batch size, ped num, feature output dim)
        lstm_featured_peds = self.lstm(featured_peds)  # output : (batch_size, ped num, hidden_layert)
        # 2. 각각 보행자 feature 와 관계 feature 비율을 이용한 중요도 파악 (Softmax)
        weights = self.ped_weighted(torch.cat((lstm_featured_peds, featured_peds)), dim=-1)
        softmaxed_weights = torch.nn.Softmax(weights)


# 3. Softmax를 기반으로 Social Force 생성

# 4. 해당 Social Force 영역과 실제 이동 위치가 비슷한 관계를 Loss 로 계산

# 5. 학습이 완료된 것을 통해 보상함수로 사용

