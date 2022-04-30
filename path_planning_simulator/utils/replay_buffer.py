import numpy as np
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"


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


class ReplayBufferTensored():
    def __init__(self, state_dim: int, action_dim: int, max_size=int(1e4)):

        self.max_size = max_size

        self.ptr = 0
        self.size = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state = torch.zeros((max_size, state_dim), device=self.device)
        self.action = torch.zeros((max_size, action_dim), device=self.device)
        self.next_state = torch.zeros((max_size, state_dim), device=self.device)
        self.reward = torch.zeros((max_size, 1), device=self.device)
        self.done_mask = torch.zeros((max_size, 1), device=self.device)

    def store(self, sample):
        s, a, r, p, d = sample
        self.state[self.ptr] = s

        self.action[self.ptr] = a
        self.next_state[self.ptr] = r
        self.reward[self.ptr] = p
        self.done_mask[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind].clone(),
            self.action[ind].clone(),
            self.next_state[ind].clone(),
            self.reward[ind].clone(),
            self.done_mask[ind].clone()
        )