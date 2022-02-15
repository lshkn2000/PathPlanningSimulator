import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from path_planning_simulator.utils.custom_state import FeaturedLSTM
from path_planning_simulator.utils.featured_state import FeaturedState


device = "cuda:0" if torch.cuda.is_available() else "cpu"
debug = False


class ReplayBuffer():
    def __init__(self, max_size=100000, batch_size=64):
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

        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        new_states = torch.from_numpy(new_states).float().to(device)
        is_terminals = torch.from_numpy(is_terminals).float().to(device)

        return states, actions, rewards, new_states, is_terminals

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        # state shape : [256, 32]
        a = self._format(state)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        action = self.max_action * torch.tanh(self.l3(a))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(self,
                 input_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 batch_size=256,
                 ):

        # self.lstm = FeaturedLSTM(input_dim=5, output_dim=input_dim, hidden_dim=input_dim)    # 보행자의 속성 [px, py, vx, vy, r]
        # self.featured_state = FeaturedState(batch_size=batch_size)
        # self.featured_state_optimizer = torch.optim.Adam(self.featured_state.parameters(), lr=3e-4)


        self.replay_buffer = ReplayBuffer(max_size=100000, batch_size=batch_size)

        self.actor = Actor(input_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(input_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def predict(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self):
        experiences = self.replay_buffer.sample()
        self.optimize_model(experiences)

    def optimize_model(self, experiences):
        self.total_it += 1
        states, actions, rewards, next_states, is_terminals = experiences

        # states = self.lstm.custom_state_for_lstm(states)
        # next_states = self.lstm.custom_state_for_lstm(next_states)
        # states = self.featured_state(states)
        # next_states = self.featured_state(next_states)

        # Optimize Critic
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.discount * (1 - is_terminals) * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            # self.featured_state_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # self.featured_state_optimizer.step()

    # def update_network(self):
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_trajectory(self, state, action, reward, next_state, is_terminal):
        experience = (state, action, reward, next_state, 1 if is_terminal else 0)
        self.replay_buffer.store(experience)

    def save(self, filename):
        torch.save({
            "online_actor_model": self.actor.state_dict(),
            "online_critic_model": self.critic.state_dict(),
            "target_actor_model": self.actor_target.state_dict(),
            "target_critic_model": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, filename if filename[-4:] == ".tar" else filename + ".tar")

    def load(self, filename):
        filename = filename if filename[-4:] == ".tar" else filename + ".tar"
        check_point = torch.load(filename)

        self.actor.load_state_dict(check_point['online_actor_model'])
        self.critic.load_state_dict(check_point["online_critic_model"])
        self.actor_target.load_state_dict(check_point["target_actor_model"])
        self.critic_target.load_state_dict(check_point["target_critic_model"])
        self.actor_optimizer.load_state_dict(check_point["actor_optimizer"])
        self.critic_optimizer.load_state_dict(check_point["critic_optimizer"])

# def save(self, filename):
# torch.save(self.critic.state_dict(), filename + "_critic")
# torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
#
# torch.save(self.actor.state_dict(), filename + "_actor")
# torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
#
# def load(self, filename):
# self.critic.load_state_dict(torch.load(filename + "_critic"))
# self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
# self.critic_target = copy.deepcopy(self.critic)
#
# self.actor.load_state_dict(torch.load(filename + "_actor"))
# self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
# self.actor_target = copy.deepcopy(self.actor)