# from IPython.display import clear_output
# %matplotlib notebook
import random
import time
import datetime
from tqdm import tqdm

import numpy as np
import gc
from torch.utils.tensorboard import SummaryWriter

from PathPlanningSimulator_new.path_planning_simulator.sim.environment import Environment
from PathPlanningSimulator_new.path_planning_simulator.sim.robot import Robot
from PathPlanningSimulator_new.path_planning_simulator.sim.obstacle import DynamicObstacle, StaticObstacle
from PathPlanningSimulator_new.path_planning_simulator.policy.linear import Linear

from PathPlanningSimulator_new.path_planning_simulator.policy.td3_new import TD3

from PathPlanningSimulator_new.path_planning_simulator.utils.pretrain.pretrain import PretrainingEnv
from PathPlanningSimulator_new.path_planning_simulator.utils.file_manager import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_sim(env, max_episodes=1, max_step_per_episode=100, render=False, random_action_episodes=100, n_warmup_batches=5, **kwargs):
    # simulation start
    start_time = time.time()
    dt = env.time_step

    # Random Seed Setting
    SEED = 425
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Info About Env
    print(env.robot.info)

    # Tensorboard Logging
    plot_log_data = SummaryWriter()

    # Simulate
    total_collision = 0
    total_goal = 0
    total_time_out = 0

    cnt = 0
    sim_episodes = tqdm(range(1, max_episodes+1))
    for i_episode in sim_episodes:
        # reset setting
        is_terminal = False
        score = 0.0
        time_step_for_ep = 0

        # train
        state = env.reset(random_position=False, random_goal=False, max_steps=max_step_per_episode)

        for t in range(1, max_step_per_episode+1):
            cnt += 1
            # Policy
            if i_episode >= random_action_episodes:
                action = env.robot.act(state)       # if cartesian : [Vx Vy] else : [W, V]
                # Action with noise
                action += np.random.normal(0.0, max_action_scale * action_noise, size=action_space)
                action = action.clip(-max_action_scale, max_action_scale)
            else:   # Action Setting with Random
                action = np.random.randn(action_space).clip(-max_action_scale, max_action_scale)

            new_state, reward, is_terminal, info = env.step(action)

            env.robot.store_trjectory(state, action, reward, new_state, is_terminal)

            state = new_state
            time_step_for_ep += 1
            score += reward

            # Train Policy
            min_samples = env.robot.policy.replay_buffer.batch_size * n_warmup_batches
            if len(env.robot.policy.replay_buffer) > min_samples:
                env.robot.policy.train()

            # Check Episode Terminal
            if is_terminal or t == max_step_per_episode:
                if info == 'Goal':
                    total_goal += 1
                elif info == 'Collision' or info == 'OutBoundary':
                    total_collision += 1
                elif info == 'TimeOut':
                    total_time_out += 1

                sim_episodes.set_postfix({'episode': i_episode, 'steps': time_step_for_ep, 'reward': score,
                                          'Total Episode': i_episode, 'Total Collision': total_collision,
                                          'Total Goal': total_goal,
                                          'Total Time Out': total_time_out, 'Success Rate': total_goal/i_episode})
                # log success rate
                plot_log_data.add_scalar('Success Rate', total_goal / i_episode, i_episode)

                gc.collect()
                break

        # save learning weights
        if i_episode % 100 == 0 and i_episode != 0:
            env.robot.policy.save("learning_data/tmp")

        # render check
        if render and i_episode % 1 == 0 and i_episode != 0 and time_step_for_ep < 120:
            env.render(path_info=True, is_plot=False)

    print('####################################')
    env.robot.policy.save("learning_data/total")
    print('####################################')

    end_time = time.time() - start_time
    plot_log_data.close()
    print("simulation operating time : {}".format(str(datetime.timedelta(seconds=end_time))))
    print("done!")


if __name__ == "__main__":
    PATH = r'/home/rvlab/PathPlanningSimulator_branch/PathPlanningSimulator_Package/PathPlanningSimulator_new/path_planning_simulator'

    # 환경 소환
    map_width = 10
    map_height = 10
    is_relative = True
    env = Environment(map_width, map_height,
                      start_rvo2=True, is_relative=is_relative, is_obstacle_sort=True,
                      safe_distance=2.0)

    # 환경 변수 설정
    time_step = 0.1                                         # real time 고려 한 시간 스텝 (s)
    max_step_per_episode = 200                              # 시뮬레이션 상에서 에피소드당 최대 스텝 수
    time_limit = max_step_per_episode                       # 시뮬레이션 스텝을 고려한 real time 제한 소요 시간
    max_episodes = 50000
    env.set_time_step_and_time_limit(time_step, time_limit)
    action_noise = 0.1

    # pretraining paramter
    pretrain_episodes = 5000
    pretraining_file_path = './utils/pretrain'
    pretraining_file_name = 'buffer_dict.pkl'
    is_pretraining = False

    # model load
    model_path = 'learning_data/tmp'

    # 로봇 소환
    # 1. 로봇 초기화
    robot = Robot(robot_name="Robot", is_relative=is_relative)
    # robot_init_position = {"px":0, "py":-2, "vx":0, "vy":0, "gx":0, "gy":4, "radius":0.2}
    robot.set_agent_attribute(px=0, py=-2, vx=0, vy=0, gx=0, gy=4, radius=0.2, v_pref=1, time_step=time_step)
    robot.set_goal_offset(0.3)  # 0.3m 범위 내에서 목적지 도착 인정

    # 장애물 소환
    # 2. 동적 장애물
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

    # 3. 정적 장애물
    st_obstacle_num = 0
    st_obstacles = [None] * st_obstacle_num
    for i in range(st_obstacle_num):
        st_obstacle = StaticObstacle()

        # 초기 소환 위치 설정
        st_obstacle.set_agent_attribute(px=-2, py=2, vx=0, vy=0, gx=0, gy=0, radius=0.3, v_pref=1, time_step=time_step)
        # 장애물을 사각형으로 설정할 것이라면
        st_obstacle.set_rectangle(width=0.3, height=0.3)

        st_obstacles[i] = st_obstacle

    # 4. 로봇 정책(행동 규칙) 세팅
    # robot state(x, y, vx, vy, gx, gy, radius) + dy_obt(x,y,vx,vy,r) + st_obt(x,y,width, height)
    basic_observation_space = 7 + (dy_obstacle_num * 5) + (st_obstacle_num * 4)

    # RL Model input dimension
    observation_space = basic_observation_space
    # Setting Robot Action space
    action_space = 2    # 이산적이라면 상,하,좌,우, 대각선 방향 총 8가지
    max_action_scale = 1    # 가속 테스트용이 아니라면 스케일은 1로 고정하는 것을 추천. 속도 정보를 바꾸려면 로봇 action 에서 직접 바꾸는 방식이 좋을 듯 하다.

    # robot_policy = Random()
    robot_policy = TD3(observation_space, action_space, max_action=max_action_scale)
    robot.set_policy(robot_policy)

    # 5. 환경에 로봇과 장애물 세팅하기
    env.set_robot(robot)

    # 환경에 장애물 정보 세팅
    for obstacle in dy_obstacles:
        env.set_dynamic_obstacle(obstacle)
    for obstacle in st_obstacles:
        env.set_static_obstacle(obstacle)


############################## PRETRAINING ####################################
    if is_pretraining:
        # Reset for pretraining
        env.reset()
        # 1) Collecting Pretrain Data
        pretrain_env = PretrainingEnv(env, time_step)

        # If pretrain data exist, Get that.
        pretrain_env.data_load(pretraining_file_path, pretraining_file_name, pretrain_episodes)

        pretrain_env.pretraining(pretrain_episodes=pretrain_episodes)

        # 학습된 모델 저장
        pretrain_env.save_model()

        print("################")
        print("Pretraining Done")
        print("################")
    else:
        pass

###################################################################################
    # 학습 가중치 가져오기
    try:
        robot.policy.load(model_path)
    except Exception as e:
        print("RL Model Not Exist! Start Learning Without it")
        pass

    run_sim(env, max_episodes=max_episodes, max_step_per_episode=max_step_per_episode, render=False,
            random_action_episodes=0, n_warmup_batches=5)