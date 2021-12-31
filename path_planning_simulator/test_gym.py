import random

import gym
import torch
import numpy as np
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from policy.dqn import DQN
from policy.ddqn import DDQN
from policy.duelingdqn import DuelingDDQN
from policy.per import PER
from policy.ddpg import DDPG
from policy.td3 import TD3
from policy.sac import SAC

from utils.plot_graph import plot_data

# 이산적인 행동에 대한 알고리즘
env = gym.make('CartPole-v1')
policy = DQN(env.observation_space.shape[0], env.action_space.n, gamma=0.98, lr=0.0005)
# policy = DDQN(env.observation_space.shape[0], env.action_space.n, gamma=0.98, lr=0.0007)
# policy = DuelingDDQN(env.observation_space.shape[0], env.action_space.n, gamma=0.98, lr=0.0007)
# policy = PER(env.observation_space.shape[0], env.action_space.n, gamma=0.98, lr=0.0007)

# 연속적인 행동에 대한 알고리즘
# env = gym.make('Pendulum-v0')
# policy = DDPG(env.observation_space.shape[0], env.action_space.shape[0], action_space_low=[-2], action_space_high=[2], gamma=0.99, lr=0.0003)
# policy = TD3(env.observation_space.shape[0], env.action_space.shape[0], action_space_low=[-2], action_space_high=[2], gamma=0.99, lr=0.0003)
# policy = SAC(env.observation_space.shape[0], env.action_space.shape[0], action_space_low=[-2], action_space_high=[2], gamma=0.99, policy_optimizer_lr=0.0005, value_optimizer_lr=0.0007, tau=0.005)

################################################################################################################

SEED = (12, 34)#, 56, 78, 90)
max_episodes = 30

total_seeds_episodes_results = []
plot_log_data = SummaryWriter()
for i, seed in enumerate(SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    episodes_result = []

    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        is_terminal = False
        score = 0.0
        time_step = 0

        n_warmup_batches = 5
        update_target_every_steps = 1 # 1 for DDPG

        while not is_terminal:
            # action = policy.training_strategy.select_action(policy.online_model, state)
            action = policy.predict(state)
            new_state, reward, is_terminal, info = env.step(action)
            policy.store_trajectory(state, action, reward, new_state, is_terminal)

            state = new_state
            time_step += 1
            score += reward

            min_samples = policy.replay_buffer.batch_size * n_warmup_batches
            if len(policy.replay_buffer) > min_samples:
                policy.train()
                # policy.train(time_step) # for TD3

            if time_step % update_target_every_steps == 0:
                policy.update_network()
                # policy.update_network(time_step, update_target_policy_every_steps=2, update_target_value_every_steps=2) # for TD3

            if is_terminal:
                print("{} episode, {} score : ".format(i_episode, score))
                gc.collect()
                break

        # stat
        episodes_result.append(score)
        # plot_log_data.add_scalar('Reward', score, i_episode)     # Tensorboard

    total_seeds_episodes_results.append(episodes_result)
    print("{} set of simulation done", i)

plot_data(np.array(total_seeds_episodes_results), smooth=100, show=True, save=False)
print("done!")
env.close()
