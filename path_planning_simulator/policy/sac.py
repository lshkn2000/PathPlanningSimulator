import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self,
                 max_size=10000,
                 batch_size=64):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size


class FCQSA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCQSA, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim + output_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u,
                             device=self.device,
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(torch.cat((x, u), dim=1)))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class FCGP(nn.Module):
    def __init__(self,
                 input_dim,
                 action_bounds,
                 log_std_min=-20,
                 log_std_max=2,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu,
                 entropy_lr=0.001):
        super(FCGP, self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = nn.Linear(input_dim,
                                     hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)

        self.nn_min = F.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = F.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=entropy_lr)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std,
                                self.log_std_min,
                                self.log_std_max)
        return x_mean, x_log_std

    def full_pass(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)

        pi_s = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)

        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log(
            (1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale_fn(torch.tanh(mean))

    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken) / (env_max - env_min)))

    def _get_actions(self, state):
        mean, log_std = self.forward(state)

        action = self.rescale_fn(torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale_fn(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(),
                                          high=self.env_max.cpu().numpy())

        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action


class SAC():
    def __init__(self, observation_space, action_space, action_space_low:list, action_space_high:list, gamma=0.99, policy_optimizer_lr=0.0005, value_optimizer_lr=0.0007, tau=0.005):
        # hyper-parameters
        self.gamma = gamma
        self.n_warmup_batches = 10
        self.value_max_grad_norm = float('inf')        # huber loss 델타 값
        self.policy_max_grad_norm = float('inf')       # huber loss 델타 값
        self.tau = tau

        # action bounds
        # env.action_space.low : 연속적 행동의 최소값 , env.action_space.high : 연속적 행동의 최대값
        if not isinstance(action_space_low, np.ndarray) and not isinstance(action_space_high, np.ndarray):
            action_space_low = np.array(action_space_low, dtype=np.float32)
            action_space_high = np.array(action_space_high, dtype=np.float32)
        bounds = action_space_low, action_space_high

        # model
        self.replay_buffer = ReplayBuffer(max_size=100000, batch_size=256)

        self.online_model = FCGP(observation_space, action_bounds=bounds, hidden_dims=(256, 256))

        self.online_value_model_a = FCQSA(observation_space, action_space, hidden_dims=(256, 256))
        self.target_value_model_a = FCQSA(observation_space, action_space, hidden_dims=(256, 256))
        self.online_value_model_b = FCQSA(observation_space, action_space, hidden_dims=(256, 256))
        self.target_value_model_b = FCQSA(observation_space, action_space, hidden_dims=(256, 256))
        self.update_value_networks(tau=1.0)

        self.policy_optimizer = optim.Adam(self.online_model.parameters(), lr=policy_optimizer_lr)
        self.value_optimizer_a = optim.Adam(self.online_value_model_a.parameters(), lr=value_optimizer_lr)
        self.value_optimizer_b = optim.Adam(self.online_value_model_b.parameters(), lr=value_optimizer_lr)

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        # policy loss
        current_actions, logpi_s, _ = self.online_model.full_pass(states)

        target_alpha = (logpi_s + self.online_model.target_entropy).detach()
        alpha_loss = -(self.online_model.logalpha * target_alpha).mean()

        self.online_model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.online_model.alpha_optimizer.step()
        alpha = self.online_model.logalpha.exp()

        current_q_sa_a = self.online_value_model_a(states, current_actions)
        current_q_sa_b = self.online_value_model_b(states, current_actions)
        current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
        policy_loss = (alpha * logpi_s - current_q_sa).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.policy_max_grad_norm)
        self.policy_optimizer.step()

        # Q loss
        ap, logpi_sp, _ = self.online_model.full_pass(next_states)
        q_spap_a = self.target_value_model_a(next_states, ap)
        q_spap_b = self.target_value_model_b(next_states, ap)
        q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
        target_q_sa = (rewards + self.gamma * q_spap * (1 - is_terminals)).detach()

        q_sa_a = self.online_value_model_a(states, actions)
        q_sa_b = self.online_value_model_b(states, actions)
        qa_loss = (q_sa_a - target_q_sa).pow(2).mul(0.5).mean()
        qb_loss = (q_sa_b - target_q_sa).pow(2).mul(0.5).mean()

        self.value_optimizer_a.zero_grad()
        qa_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_a.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer_a.step()

        self.value_optimizer_b.zero_grad()
        qb_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model_b.parameters(),
                                       self.value_max_grad_norm)
        self.value_optimizer_b.step()

        ###############????????????????????????????????????###########################
        # self.policy_optimizer.zero_grad()
        # policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),
        #                                self.policy_max_grad_norm)
        # self.policy_optimizer.step()
        ##############################################################################

    def update_value_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model_a.parameters(),
                                  self.online_value_model_a.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_value_model_b.parameters(),
                                  self.online_value_model_b.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def update_network(self):
        self.update_value_networks()

    def predict(self, state):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        if len(self.replay_buffer) < min_samples:
            action = self.online_model.select_random_action(state)
        else:
            action = self.online_model.select_action(state)
        return action

    def store_trajectory(self, state, action, reward, next_state, is_terminal):
        experience = (state, action, reward, next_state, float(is_terminal))
        self.replay_buffer.store(experience)

    def train(self):
        experiences = self.replay_buffer.sample()
        experiences = self.online_value_model_a.load(experiences)
        self.optimize_model(experiences)

    def save(self, filename):
        torch.save({
            "online_model": self.online_model.state_dict(),
            "online_value_model_a": self.online_value_model_a.state_dict(),
            "online_value_model_b": self.online_value_model_b.state_dict(),
            "target_value_model_a": self.target_value_model_a.state_dict(),
            "target_value_model_b": self.target_value_model_b.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer_a": self.value_optimizer_a.state_dict(),
            "value_optimizer_b": self.value_optimizer_b.state_dict(),

        }, filename if filename[-4:] == ".tar" else filename + ".tar")

    def load(self, filename):
        filename = filename if filename[-4:] == ".tar" else filename + ".tar"
        check_point = torch.load(filename)

        self.online_model.load_state_dict(check_point['online_model'])

        self.online_value_model_a.load_state_dict(check_point["online_value_model_a"])
        self.online_value_model_b.load_state_dict(check_point["online_value_model_b"])
        self.target_value_model_a.load_state_dict(check_point["target_value_model_a"])
        self.target_value_model_b.load_state_dict(check_point["target_value_model_b"])

        self.policy_optimizer.load_state_dict(check_point["policy_optimizer"])
        self.value_optimizer_a.load_state_dict(check_point["value_optimizer_a"])
        self.value_optimizer_b.load_state_dict(check_point["value_optimizer_b"])




