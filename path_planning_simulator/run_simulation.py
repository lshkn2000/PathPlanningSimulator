# from IPython.display import clear_output
# %matplotlib notebook
import random

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
from utils.plot_graph import plot_data


def run_sim(env, max_episodes=1, max_step_per_episode=50, render=True, seed_num=1, n_warmup_batches=5, update_target_every_steps=1, **kwargs):
    SEED = [random.randint(1, 100) for _ in range(seed_num)]
    dt = env.time_step

    # 각 로봇, 동적 장애물 행동 취하기
    # 에피소드 실행
    print(env.robot.info)
    print(env.dy_obstacles[0].info)

    total_seeds_episodes_results = []
    plot_log_data = SummaryWriter()
    for i, seed in enumerate(SEED):
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
            update_target_every_steps = update_target_every_steps # 1 for DDPG

            for t in range(max_step_per_episode):
                action, discrete_action_index = env.robot.act(state)  # action : (vx, vy)
                new_state, reward, is_terminal, info = env.step(action)
                if env.robot.is_discrete_actions:
                    env.robot.policy.store_trajectory(state, discrete_action_index, reward, new_state, is_terminal)
                else:
                    env.robot.policy.store_trajectory(state, action, reward, new_state, is_terminal)

                state = new_state
                time_step_for_ep += 1
                score += reward

                min_samples = env.robot.policy.replay_buffer.batch_size * n_warmup_batches
                if len(env.robot.policy.replay_buffer) > min_samples:
                    env.robot.policy.train()
                    # policy.train(time_step_for_ep) # for TD3

                if time_step_for_ep % update_target_every_steps == 0:
                    env.robot.policy.update_network()
                    # policy.update_network(time_step_for_ep, update_target_policy_every_steps=2, update_target_value_every_steps=2) # for TD3

                if is_terminal:
                    print("{} episode, {} steps, {} score".format(i_episode, time_step_for_ep, score))
                    gc.collect()
                    break

            # stat
            episodes_result.append(score)
            plot_log_data.add_scalar('Reward for seed {}'.format(i), score, i_episode)     # Tensorboard

            # render check
            if render:
                env.render(path_info=False)

        total_seeds_episodes_results.append(episodes_result)

        print('####################################')
        torch.save(env.robot.policy.online_model.state_dict(), 'learning_data/test_20211131.pt')
        print("{} set of simulation done".format(i+1))
        print('####################################')

    plot_data(np.array(total_seeds_episodes_results), smooth=100, show=True, save=False)
    print("done!")


if __name__ == "__main__":
    # 환경 소환
    env = Environment(start_rvo2=True)

    # 환경 변수 설정
    time_step = 0.1                                         # real time 고려 한 시간 스텝 (s)
    max_step_per_episode = 10000                            # 시뮬레이션 상에서 에피소드당 최대 스텝 수
    time_limit = int(max_step_per_episode * time_step)      # 시뮬레이션 스텝을 고려한 real time 제한 소요 시간
    env.set_time_step_and_time_limit(time_step, time_limit)

    # 로봇 소환
    discrete_action_space = 8 # continuous action space 이면 None 또는 주석 처리!
    robot = Robot(discrete_action_space=discrete_action_space)
    # robot_init_position = {"px":0, "py":-2, "vx":0, "vy":0, "gx":0, "gy":4, "radius":0.2}
    robot.set_agent_attribute(px=0, py=-2, vx=0, vy=0, gx=0, gy=4, radius=0.2, v_pref=1, time_step=time_step)

    # 장애물 소환
    # 동적 장애물
    dy_obstacle_num = 10
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

    # 정적 장애물
    st_obstacle_num = 1
    st_obstacles = [None] * st_obstacle_num
    for i in range(st_obstacle_num):
        st_obstacle = StaticObstacle()

        # 초기 소환 위치 설정
        st_obstacle.set_agent_attribute(px=-2, py=2, vx=0, vy=0, gx=0, gy=0, radius=0.3, v_pref=1, time_step=time_step)
        # 장애물을 사각형으로 설정할 것이라면
        st_obstacle.set_rectangle(width=0.3, height=0.3)

        st_obstacles[i] = st_obstacle

    # 로봇 정책(행동 규칙) 세팅
    observation_space = 7 + (dy_obstacle_num * 5) + (st_obstacle_num * 4) # robot state(x, y, vx, vy, gx, gy, radius) + dy_obt(x,y,vx,vy,r) + st_obt(x,y,width, height)
    action_space = discrete_action_space    # 상,하,좌,우, 대각선 방향 총 8가지
    robot_policy = DQN(observation_space, action_space, gamma=0.98, lr=0.0005)
    # robot_policy = Random()
    robot.set_policy(robot_policy)

    # 환경에 로봇과 장애물 세팅하기
    env.set_robot(robot)

    # 클래스 담긴 리스트를 넘겨줄지 아니면 클래스 개별로 넘겨줄지는 효율적인 것을 고려해서 수정할 것
    for obstacle in dy_obstacles:
        env.set_dynamic_obstacle(obstacle)
    for obstacle in st_obstacles:
        env.set_static_obstacle(obstacle)

    run_sim(env, max_episodes=10000, max_step_per_episode=max_step_per_episode, render=True, seed_num=3, n_warmup_batches=5, update_target_every_steps=1)