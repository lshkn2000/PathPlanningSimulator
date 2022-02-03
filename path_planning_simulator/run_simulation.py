# from IPython.display import clear_output
# %matplotlib notebook
import os
import random
import time
import datetime
import pickle
import collections

import torch
import numpy as np
import gc
from torch.utils.tensorboard import SummaryWriter

from sim.environment import Environment
from sim.robot import Robot
from sim.obstacle import DynamicObstacle, StaticObstacle
from policy.random import Random
from policy.linear import Linear
from policy.dqn import DQN
from policy.sac import SAC
# from policy.td3 import TD3
from policy.td3_new import TD3
from utils.plot_graph import plot_data

# from utils.pretrained import PretrainedSim


def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def run_sim(env, max_episodes=1, max_step_per_episode=50, render=True, seed_num=1, n_warmup_batches=5, update_target_interval=1, **kwargs):
    # simulation start
    start_time = time.time()
    timestr = time.strftime("%Y%m%d_%H%M%S")    # 학습 데이터 저장용

    SEED = [random.randint(1, 100) for _ in range(seed_num)]
    dt = env.time_step

    # 각 로봇, 동적 장애물 행동 취하기
    # 에피소드 실행
    print(env.robot.info)
    # print(env.dy_obstacles[0].info)

    total_seeds_episodes_results = []
    plot_log_data = SummaryWriter()

    total_collision = 0
    total_goal = 0
    total_time_out = 0

    for i_seed, seed in enumerate(SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        episodes_result = []

        for i_episode in range(1, max_episodes+1):
            state = env.reset(random_position=False, random_goal=False, max_steps=max_step_per_episode)
            is_terminal = False
            score = 0.0
            time_step_for_ep = 0

            n_warmup_batches = n_warmup_batches
            update_target_interval = update_target_interval # 1 for DDP

            for t in range(1, max_step_per_episode+1):
                if i_episode > 50:
                    action, discrete_action_index = env.robot.act(state)  # action : (vx, vy)
                    action += np.random.normal(0.0, 1.0, size=action_space)
                    action = action.clip(-max_action_scale, max_action_scale)
                else:
                    if env.robot.is_discrete_actions:
                        discrete_action_index = np.random.randint(action_space)
                    else:
                        action = np.random.randn(action_space).clip(-max_action_scale, max_action_scale)
                new_state, reward, is_terminal, info = env.step(action)
                if env.robot.is_discrete_actions:
                    env.robot.store_trjectory(state, discrete_action_index, reward/10, new_state, is_terminal)
                else:
                    env.robot.store_trjectory(state, action, reward/10, new_state, is_terminal)

                state = new_state
                time_step_for_ep += 1
                score += reward

                min_samples = env.robot.policy.replay_buffer.batch_size * n_warmup_batches
                if len(env.robot.policy.replay_buffer) > min_samples:
                    env.robot.policy.train()
                    # env.robot.policy.train(time_step_for_ep) # for TD3

                if time_step_for_ep % update_target_interval == 0:
                    env.robot.policy.update_network()
                    # env.robot.policy.update_network(time_step_for_ep, update_target_policy_every_steps=2, update_target_value_every_steps=2) # for TD3

                if is_terminal or t == max_step_per_episode:
                    print("{} seeds {} episode, {} steps, {} reward".format(i_seed, i_episode, time_step_for_ep, score))
                    if info == 'Goal':
                        total_goal += 1
                    elif info == 'Collision' or info == 'OutBoundary':
                        total_collision += 1
                    elif info == 'TimeOut':
                        total_time_out += 1

                    print(
                        "Total Episode : {:5} , Total Collision : {:5f} , Total Goal : {:5f} , Total Time Out : {:5f}, Success Rate : {:.4f}".format(
                            i_episode, total_collision, total_goal, total_time_out, total_goal / i_episode))
                    gc.collect()
                    break

            # stat
            episodes_result.append(score)

            # log learning weights
            plot_log_data.add_scalar('Reward for seed {}'.format(i_seed), score, i_episode)     # Tensorboard

            # save learning weights
            if i_episode % 100 == 0 and i_episode != 0:
                env.robot.policy.save("learning_data/tmp")

            # render check
            if render and i_episode % 50 == 0 and i_episode != 0 and time_step_for_ep < 120:
                env.render(path_info=True, is_plot=False)

        total_seeds_episodes_results.append(episodes_result)

        print('####################################')
        env.robot.policy.save("learning_data/total")
        print("{} set of simulation done".format(seed_num))
        print('####################################')

    plot_data(np.array(total_seeds_episodes_results), smooth=100, show=True, save=True)
    end_time = time.time() - start_time
    plot_log_data.close()
    print("simulation operating time : {}".format(str(datetime.timedelta(seconds=end_time))))
    print("done!")


if __name__ == "__main__":
    make_directories("learning_data/reward_graph")
    make_directories("learning_data/video")

    # 환경 소환
    env = Environment(start_rvo2=True)

    # 환경 변수 설정
    time_step = 0.1                                         # real time 고려 한 시간 스텝 (s)
    max_step_per_episode = 200                              # 시뮬레이션 상에서 에피소드당 최대 스텝 수
    time_limit = max_step_per_episode                       # 시뮬레이션 스텝을 고려한 real time 제한 소요 시간
    max_episodes = 10000
    env.set_time_step_and_time_limit(time_step, time_limit)
    seed_num = 3

    # 로봇 소환
    # 1. 행동이 이산적인지 연속적인지 선택
    # 2. 로봇 초기화
    is_discrete_action_space = None # continuous action space 이면 None
    robot = Robot(discrete_action_space=is_discrete_action_space, is_holomonic=True, robot_name="Robot")
    # robot_init_position = {"px":0, "py":-2, "vx":0, "vy":0, "gx":0, "gy":4, "radius":0.2}
    robot.set_agent_attribute(px=0, py=-2, vx=0, vy=0, gx=0, gy=4, radius=0.2, v_pref=1, time_step=time_step)

    # 장애물 소환
    # 3. 동적 장애물
    dy_obstacle_num = 5
    dy_obstacles = [None] * dy_obstacle_num
    for i in range(dy_obstacle_num):
        dy_obstacle = DynamicObstacle()

        # 초기 소환 위치 설정
        # 이것도 함수로 한번에 설정할 수 있도록 변경할 것
        dy_obstacle.set_agent_attribute(px=2, py=2, vx=0, vy=0, gx=-2, gy=-2, radius=0.3, v_pref=1, time_step=time_step)
        # 동적 장애물 정책 세팅
        dy_obstacle_policy = Linear()
        dy_obstacle.set_policy(dy_obstacle_policy)

        dy_obstacles[i] = dy_obstacle

    # 4. 정적 장애물
    st_obstacle_num = 0
    st_obstacles = [None] * st_obstacle_num
    for i in range(st_obstacle_num):
        st_obstacle = StaticObstacle()

        # 초기 소환 위치 설정
        st_obstacle.set_agent_attribute(px=-2, py=2, vx=0, vy=0, gx=0, gy=0, radius=0.3, v_pref=1, time_step=time_step)
        # 장애물을 사각형으로 설정할 것이라면
        st_obstacle.set_rectangle(width=0.3, height=0.3)

        st_obstacles[i] = st_obstacle

    # 5. 로봇 정책(행동 규칙) 세팅
    observation_space = 7 + (dy_obstacle_num * 5) + (st_obstacle_num * 4) # robot state(x, y, vx, vy, gx, gy, radius) + dy_obt(x,y,vx,vy,r) + st_obt(x,y,width, height)
    # observation_space = 17
    # 로봇의 action space 설정
    action_space = 2    # 이산적이라면 상,하,좌,우, 대각선 방향 총 8가지
    max_action_scale = 1    # 가속 테스트용이 아니라면 스케일은 1로 고정하는 것을 추천. 속도 정보를 바꾸려면 로봇 action 에서 직접 바꾸는 방식이 좋을 듯 하다.
    # robot_policy = Random()
    # robot_policy = DQN(observation_space, action_space, gamma=0.98, lr=0.0005)
    # robot_policy = SAC(observation_space, action_space, action_space_low=[-1, -1], action_space_high=[1, 1], gamma=0.99, policy_optimizer_lr=0.0005, value_optimizer_lr=0.0007, tau=0.005)
    # robot_policy = TD3(observation_space, action_space, action_space_low=[-max_action_scale, -max_action_scale],
    #                    action_space_high=[max_action_scale, max_action_scale], gamma=0.99, lr=0.0003)
    robot_policy = TD3(observation_space, action_space, max_action=max_action_scale)
    robot.set_policy(robot_policy)
    # 학습 가중치 가져오기
    # robot.policy.load('learning_data/tmp')

    # 환경에 로봇과 장애물 세팅하기
    env.set_robot(robot)

    # 클래스 담긴 리스트를 넘겨줄지 아니면 클래스 개별로 넘겨줄지는 효율적인 것을 고려해서 수정할 것
    for obstacle in dy_obstacles:
        env.set_dynamic_obstacle(obstacle)
    for obstacle in st_obstacles:
        env.set_static_obstacle(obstacle)

    run_sim(env, max_episodes=max_episodes, max_step_per_episode=max_step_per_episode, render=False, seed_num=seed_num, n_warmup_batches=5, update_target_interval=2)