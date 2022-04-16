# from IPython.display import clear_output
# %matplotlib notebook
import os
import random
import time
import datetime
import pickle
import collections
import cv2
import torchvision.utils
from PIL import Image
import matplotlib.pyplot as plt

import numpy
from tqdm import tqdm

import torch
import torch.optim as optim
import numpy as np
import gc
from torchvision.utils import save_image
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
from utils.replay_buffer import ReplayBuffer

from utils.pretrained import PretrainedSim
from utils.pretrain import PretrainedSim
from utils.world_model.VAE import VAE
from path_planning_simulator.utils.state_function_engineering.grid_based_state_model import GridBasedState
from path_planning_simulator.utils.world_model.VAE.make_dataset import make_file_list, ImageTransform, ImgDataset
from path_planning_simulator.utils.world_model.VAE.train_cnn_vae import check_vae_model, get_model_latent

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

#############################################################################################




###############################################################################################

def run_sim(env, max_episodes=1, max_step_per_episode=100, render=False, seed_num=1, n_warmup_batches=5, **kwargs):
    # simulation start
    start_time = time.time()
    # Random Seed Setting
    SEED = [random.randint(1, 100) for _ in range(seed_num)]
    dt = env.time_step

    # Info About Env
    print(env.robot.info)
    # print(env.dy_obstacles[0].info)

    # Tensorboard Logging
    plot_log_data = SummaryWriter()

    # offline learning Buffer setting
    offline_episodes = 1000
    offline_replay_buffer = ReplayBuffer(batch_size=64)
    tmp_trajectories = collections.deque(maxlen=max_step_per_episode)

    for i_seed, seed in enumerate(SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        total_collision = 0
        total_goal = 0
        total_time_out = 0

        # Start episode
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
                if i_episode >= 0:
                    action = env.robot.act(state)       # if cartesian : [Vx Vy] else : [W, V]
                else:   # Action Setting with Random
                    action = np.random.randn(action_space).clip(-max_action_scale, max_action_scale)

                new_state, reward, is_terminal, info = env.step(action)

                z = env.robot.make_encoding_state(state)
                nz = env.robot.make_encoding_state(new_state)

                ######## test vae model########
                # z, recon1 = env.robot.make_encoding_state(state)
                # nz, recon2 = env.robot.make_encoding_state(new_state)
                # recon_test_path = os.path.join(PATH, 'vae_z_recon_test')
                # tf_test = ImageTransform()
                # grid_map_state = make_grid_img.grid_based_state_function(new_state, robot_detection_scope_radius,
                #                                                          detection_scope_resolution, img_plot_size)
                # # cv2.imwrite(recon_test_path + r'/' + str(cnt) + '.png', grid_map_state)
                # grid_map_state = tf_test(grid_map_state)
                # grid_map_state = torch.Tensor(grid_map_state).unsqueeze(0)
                # grid_map_state = grid_map_state.data.cpu()
                # recon1 = recon1.data.cpu()
                # recon2 = recon2.data.cpu()
                # print(grid_map_state.shape)
                # print(recon1.shape)
                # check_z = torch.cat([grid_map_state, recon1, recon2])
                # torchvision.utils.save_image(check_z.data.cpu(), recon_test_path+f'/{cnt}.png')
                ################################


                env.robot.store_trjectory(z, action, reward, nz, is_terminal)
                tmp_trajectories.append((z, action, reward, nz, is_terminal))

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
                        for experience in tmp_trajectories:
                            offline_replay_buffer.store(experience)
                    elif info == 'Collision' or info == 'OutBoundary':
                        total_collision += 1
                        tmp_trajectories.clear()
                    elif info == 'TimeOut':
                        total_time_out += 1
                        tmp_trajectories.clear()

                    sim_episodes.set_postfix({'seeds': i_seed, 'episode': i_episode, 'steps': time_step_for_ep, 'reward': score,
                                              'Total Episode': i_episode, 'Total Collision': total_collision, 'Total Goal': total_goal,
                                              'Total Time Out': total_time_out, 'Success Rate': total_goal/i_episode})
                    # log success rate
                    plot_log_data.add_scalar('Success Rate', total_goal / i_episode, i_episode)

                    gc.collect()
                    break

            # log score (=Total reward)
            plot_log_data.add_scalar('Reward for seed {}'.format(i_seed), score, i_episode)     # Tensorboard

            # save learning weights
            if i_episode % 100 == 0 and i_episode != 0:
                env.robot.policy.save("learning_data/tmp")

            # render check
            if render and i_episode % 1 == 0 and i_episode != 0 and time_step_for_ep < 120:
                env.render(path_info=True, is_plot=False)

        # Offline Learning
        for _ in tqdm(range(offline_episodes), desc='Offline Learning'):
            offline_experiences = offline_replay_buffer.sample()
            env.robot.policy.optimize_model(offline_experiences)

        print('####################################')
        env.robot.policy.save("learning_data/total")
        print("{} set of simulation done".format(seed_num))
        print('####################################')

    ##################### Validation #####################
    print("######## VALIDATION START ########")

    # set policy as evaluation mode
    env.robot.policy.eval()

    val_total_goal = 0
    val_total_collision = 0
    val_total_time_out = 0
    val_episodes = tqdm(range(1, 100+1), desc="Validation")
    for j_episode in val_episodes:
        val_is_terminal = False

        val_state = env.reset(random_position=False, random_goal=False, max_steps=max_step_per_episode)

        for t in range(1, max_step_per_episode + 1):
            val_action = env.robot.act(val_state)
            val_new_state, val_reward, val_is_terminal, val_info = env.step(val_action)

            val_state = val_new_state

            if val_is_terminal:
                if val_info == 'Goal':
                    val_total_goal += 1
                elif val_info == 'Collision' or val_info == 'OutBoundary':
                    val_total_collision += 1
                elif val_info == 'TimeOut':
                    val_total_time_out += 1

                val_episodes.set_postfix(
                    {'Total Episode': j_episode, 'Total Collision': val_total_collision, 'Total Goal': val_total_goal,
                     'Total Time Out': val_total_time_out, 'Success Rate': val_total_goal / j_episode})

                gc.collect()
                break
    print("######## VALIDATION DONE ########")

    end_time = time.time() - start_time
    plot_log_data.close()
    print("simulation operating time : {}".format(str(datetime.timedelta(seconds=end_time))))
    print("done!")


if __name__ == "__main__":
    PATH = r'/home/rvlab/PathPlanningSimulator_branch/PathPlanningSimulator_new_worldcoord_2/path_planning_simulator'

    make_directories("learning_data/reward_graph")
    make_directories("learning_data/video")
    make_directories("vae_ckpts")

    # 환경 소환
    map_width = 10
    map_height = 10
    env = Environment(map_width, map_height, start_rvo2=True)

    # 환경 변수 설정
    time_step = 0.1                                         # real time 고려 한 시간 스텝 (s)
    max_step_per_episode = 200                              # 시뮬레이션 상에서 에피소드당 최대 스텝 수
    time_limit = max_step_per_episode                       # 시뮬레이션 스텝을 고려한 real time 제한 소요 시간
    max_episodes = 10000
    env.set_time_step_and_time_limit(time_step, time_limit)
    seed_num = 1
    action_noise = 0.1

    pretrain_episodes = 100000

    # CNN VAE
    is_vae = True
    vae_img_channels = 3
    vae_z_dim = 32
    vae_n_epochs = 500
    train_dataset_split_percentage = 0.7

    # vae hyperparameter
    # vae_hparameter = {"max_epochs": 10, "learning_rate": 0.002, "kld_weight":0.0001, "batch_size": 64}
    # vae_hidden_dim = [128, 64, 32]

    # 로봇 소환
    # 1. 행동이 이산적인지 연속적인지 선택
    # 2. 로봇 초기화
    robot = Robot(cartesian=True, robot_name="Robot", state_engineering="VAE") # ["Basic", "GridMap"]
    # robot_init_position = {"px":0, "py":-2, "vx":0, "vy":0, "gx":0, "gy":4, "radius":0.2}
    robot.set_agent_attribute(px=0, py=-2, vx=0, vy=0, gx=0, gy=4, radius=0.2, v_pref=1, time_step=time_step)
    robot.set_goal_offset(0.3)  # 0.3m 범위 내에서 목적지 도착 인정

    # 장애물 소환
    # 3. 동적 장애물
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
    # robot state(x, y, vx, vy, gx, gy, radius) + dy_obt(x,y,vx,vy,r) + st_obt(x,y,width, height)
    if is_vae:
        raw_observation_space = vae_z_dim
    else:
        raw_observation_space = 7 + (dy_obstacle_num * 5) + (st_obstacle_num * 4)

    # RL Model input dimension
    observation_space = raw_observation_space
    # Setting Robot Action space
    action_space = 2    # 이산적이라면 상,하,좌,우, 대각선 방향 총 8가지
    max_action_scale = 1    # 가속 테스트용이 아니라면 스케일은 1로 고정하는 것을 추천. 속도 정보를 바꾸려면 로봇 action 에서 직접 바꾸는 방식이 좋을 듯 하다.

    # robot_policy = Random()
    robot_policy = TD3(observation_space, action_space, max_action=max_action_scale)
    robot.set_policy(robot_policy)

    # 환경에 로봇과 장애물 세팅하기
    env.set_robot(robot)

    # 환경에 장애물 정보 세팅
    for obstacle in dy_obstacles:
        env.set_dynamic_obstacle(obstacle)
    for obstacle in st_obstacles:
        env.set_static_obstacle(obstacle)

    #####PRETRAINING#####
    # Reset for pretraining
    env.reset()
    # 경험 데이터 (정답 데이터) 저장용
    pretrained_replay_buffer = collections.deque(maxlen=1000000)

    # pretrain (+ VAE) 학습
    # 1) Collecting Pretrain Data
    pretrain_env = PretrainedSim(env, time_step)

    # If pretrain data exist, Get that.
    PRETRAIN_BUFFER_PATH = 'vae_ckpts/simulation_buffer_dict.pkl'
    if os.path.isfile(PRETRAIN_BUFFER_PATH):
        print("Found Pretrain Data Buffer")
        with open(PRETRAIN_BUFFER_PATH, 'rb') as f:
            buffer_dict = pickle.load(f)
        pretrain_env.set_pretrain_replay_buffer(buffer_dict["pretrain"])
        pretrain_env.set_vae_state_replay_buffer(buffer_dict['vae'])
    else:
        # 없다면 pretrain data 수집하기
        for i in tqdm(range(pretrain_episodes), desc="PreTrain Data Collecting"):  # episode
            pretrain_env.data_collection()
        # 저장하기
        pretrain_buffer = pretrain_env.get_pretrained_replay_buffer()
        vae_buffer = pretrain_env.get_vae_state_replay_buffer()
        buffer_dict = {"pretrain": pretrain_buffer, "vae": vae_buffer}
        with open(PRETRAIN_BUFFER_PATH, 'wb') as f:
            pickle.dump(buffer_dict, f)

    ############################Model based Learning#####################################
    print("get pretrain data")
    # 1. Learning VAE model
    #     1) GET Dataset
    #        1) - 1 : Get Trajectories
    if os.path.isfile(PRETRAIN_BUFFER_PATH):
        with open(PRETRAIN_BUFFER_PATH, 'rb') as f:
            buffer_dict = pickle.load(f)
            pretrain_dataset = buffer_dict["pretrain"]

    print("make img")
    #        1) - 2 : Make Img Dataset
    # make img dataset folder
    img_save_dir = os.path.join(PATH, 'vae_ckpts/img_dataset')
    make_directories(img_save_dir)
    # # make grid img
    make_grid_img = GridBasedState(is_relative=False)
    robot_detection_scope_radius = 10
    detection_scope_resolution = 0.1
    img_plot_size = (10, 10)

    ##### MAKE IMG ANG SAVE #####
    # for idx, pre_data in enumerate(tqdm(pretrain_dataset, desc="Make Img Dataset From Pretrain Data")):
    #     state = pre_data[0]
    #
    #     grid_map_state = make_grid_img.grid_based_state_function(state, robot_detection_scope_radius, detection_scope_resolution, img_plot_size)
    #     cv2.imwrite(img_save_dir + r'/' + str(idx) + '.png', grid_map_state)
    # print("IMG SAVE DONE")


    print("get vae")
    #     2) GET VAE Model
    # learning model setting
    vae_model_dir = os.path.join(PATH, r'utils/world_model/VAE/vae_models')
    vae_model = VAE.ConvVAE(img_channels=vae_img_channels, latent_size=vae_z_dim).to(device=device)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # VAE 모델 저장 폴더 확인 및 기존 모델 가져오기
    if not os.path.exists(vae_model_dir):
        os.mkdir(vae_model_dir)

    load_vae_model = os.path.join(vae_model_dir, 'best.tar')
    if os.path.exists(load_vae_model):
        state = torch.load(load_vae_model)
        vae_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])

    print("get img")
    #     3) Make Img dataloader
    vae_train_img_list = make_file_list(img_save_dir)
    img_dataset = ImgDataset(vae_train_img_list, ImageTransform())

    len_train_dataset = int(len(img_dataset) * train_dataset_split_percentage)
    len_test_dataset = len(img_dataset) - len_train_dataset
    train_dataset, test_dataset = torch.utils.data.random_split(img_dataset, [len_train_dataset, len_test_dataset])
    # data loader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)
    total_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=1, shuffle=False)
    # check data loader
    # fixed_x = next(iter(train_dataloader))
    # save_image(fixed_x, 'test_dataloader_img.png')
    # exit()

    #        4) Train VAE
    cur_best_model = None
    for epoch in range(1, vae_n_epochs + 1):
        vae_train_loss = 0
        for batch_idx, data_img in enumerate(train_dataloader):
            data_img = data_img.to(device)
            recon_x, mu, log_sigma, z = vae_model(data_img)
            loss, BCE, KLD = vae_model.loss_function(recon_x, data_img, mu, log_sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            vae_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, BCE : {:3f}'.format(
                    epoch, batch_idx * len(data_img), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader),
                           loss.item() / len(data_img), BCE.item() / len(data_img)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, vae_train_loss / len(train_dataloader.dataset)))

        # check point
        best_filename = os.path.join(vae_model_dir, 'best.tar')

        torch.save({
            'state_dict': vae_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, best_filename)

    # test vae model
    print("test vae model")
    check_vae_model(vae_model, test_dataset)

    exit()

    # Set VAE Model To Robot
    env.robot.set_vae_model(vae_model)
    env.robot.set_state_img_param(ImageTransform(), robot_detection_scope_radius, detection_scope_resolution, img_plot_size)

    # Get Latent Dataset
    latent_dataset_dir = os.path.join(PATH, r'utils/world_model/VAE/vae_models/latent_dataset.pkl')
    # get_model_latent(latent_dataset_dir, vae_model, total_dataloader)

    # 2. Learning MDN LSTM Model
    #     1) GET MDN LSTM Model
    #
    #     2) Get Dataset
    #
    #     3) Train MDN LSTM


################################################################################################
    # Pretrain RL Model
    # 2) CASE 1. pretrain 학습
    # pretrain_env.pretrain_with_vae_latent(latent_dataset_dir=latent_dataset_dir, pretrain_episodes=pretrain_episodes)
    # 2) CASE 2. Training VAE Model
    # vae_model, vae_normalizer = pretrain_env.trainVAE(input_dim=raw_observation_space,
    #                                                   latent_dim=observation_space,
    #                                                   hidden_dim=vae_hidden_dim,
    #                                                   **vae_hparameter)
    # vae_model.eval()
    # pretrain_env.pretrain(vae_model=vae_model, vae_normalizer=vae_normalizer)

    #####PRETRAINING Done#####

    # 학습된 모델 저장
    # pretrain_env.save_model()

    print("################")
    print("Pretraining Done")
    print("################")

    # 학습 가중치 가져오기
    robot.policy.load('learning_data/tmp')

    run_sim(env, max_episodes=max_episodes, max_step_per_episode=max_step_per_episode, render=False, seed_num=seed_num, n_warmup_batches=5)