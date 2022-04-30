import os
import collections
import random
import pickle
from tqdm import tqdm
from sklearn.preprocessing import normalize, Normalizer

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
import matplotlib.lines as mlines

import rvo2

from PathPlanningSimulator_new.path_planning_simulator.utils.calculations import *
from PathPlanningSimulator_new.path_planning_simulator.utils.file_manager import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class PretrainingEnv(object):
    def __init__(self, env, time_step, max_step=200, max_speed=1):
        self.sim = None
        self.robot_position = None
        self.target_norm = None
        self.action_noise = 0.1

        self.env = env
        self.is_relative = self.env.is_relative
        self.is_obstacle_sort = self.env.is_obstacle_sort

        self.robot = self.env.robot
        self.dy_obstacles_list = self.env.dy_obstacles_list
        self.robot_with_dy_obstacles_list = [self.robot] + self.dy_obstacles_list

        self.robot_orientation = 0      # 월드 좌표계에서의 로봇의 자세
        self.next_robot_orientation = 0     # 월드 좌표계에서의 로봇의 자세
        self.relative_delta_robot_orientation = 0   # 로봇 좌표계에서의 로봇의 자세 변화량을 계산하기 위한 용도

        self.max_steps = max_step
        self.max_speed = max_speed
        self.time_step = time_step
        self.params = {"neighborDist": 10, "maxNeighbors": 20, "timeHorizon": 5, "timeHorizonObst": 5}

        self.episode_cnt = 0

        self.pretraied_replay_buffer = collections.deque()

    def set_pretrain_replay_buffer(self, buffer):
        self.pretraied_replay_buffer = buffer

    def get_pretrained_replay_buffer(self):
        return self.pretraied_replay_buffer

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def save_model(self):
        self.env.robot.policy.save("learning_data/tmp")

    def data_load(self, path, file_name='buffer_dict.pkl', pretrain_episodes=5000):
        PRETRAIN_BUFFER_PATH = path
        make_directories(PRETRAIN_BUFFER_PATH)

        File = PRETRAIN_BUFFER_PATH + '/' + file_name
        if os.path.isfile(File):
            print("Found Pretrain Data Buffer")
            with open(File, 'rb') as f:
                buffer_dict = pickle.load(f)
            self.set_pretrain_replay_buffer(buffer_dict["pretrain"])
        else:
            # 없다면 pretrain data 수집하기
            for i in tqdm(range(pretrain_episodes), desc="PreTrain Data Collecting"):  # episode
                self.data_collection()
            # 저장하기
            pretrain_buffer = self.get_pretrained_replay_buffer()
            buffer_dict = {"pretrain": pretrain_buffer}
            with open(File, 'wb') as f:
                pickle.dump(buffer_dict, f)

    def reset(self, random_robot_spawn=True):
        self.env.reset()
        self.dy_obstacles_list = self.env.dy_obstacles_list
        if random_robot_spawn:
            self.robot.px, self.robot.py = random.uniform(-4, 4), random.uniform(-5, -3)
        else:
            self.robot = self.env.robot

        # set robot init angle
        self.robot_orientation = 0
        if (self.robot.gy - self.robot.py) >= 0:
            self.robot_orientation = np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px)
        else:
            self.robot_orientation = 2 * np.pi + np.arctan2(self.robot.gy - self.robot.py, self.robot.gx - self.robot.px)

        self.robot_with_dy_obstacles_list = [self.robot] + self.dy_obstacles_list
        self.target_norm = None

    def reward_function(self, state):
        # shape : (robot+obstacle Num, 5)
        # reward for robot position

        robot_position = state[0][:2]
        robot_velocity = state[0][2:4]
        robot_radius = state[0][-1]

        # 0 time reward
        reward = -0.05

        # 1. reward for distance
        target_distance_vector = (robot_position[0] - self.robot.goal[0], robot_position[1] - self.robot.goal[1])
        target_norm = np.linalg.norm(target_distance_vector)

        if self.target_norm is None:
            self.target_norm = target_norm

        # 목적지로의 이동 변화가 많을 수록 큰 보상
        delta_reward = lambda x: np.tanh(x) if x > 0 else np.tanh(0.9 * x)
        reward += delta_reward(self.target_norm - target_norm)

        self.target_norm = target_norm

        reach_goal = np.linalg.norm(
            robot_position - np.array(self.robot.goal)) < self.robot.radius + self.robot.goal_offset  # offset

        # 2. reward for terminal
        if reach_goal:
            reward += 10
            done = True
            info = "Goal"
            print('goal!')
            self.target_norm = None

        else:
            reward += 0
            done = False
            info = None

        return reward, done

    def pretrain_(self):
        self.reset()

        # obstacle_num = len(self.dy_obstacles_list)

        # 로봇, 장애물의 위치 정보 초기화
        self.sim = rvo2.PyRVOSimulator(self.time_step,
                                       self.params['neighborDist'],
                                       self.params['maxNeighbors'],
                                       self.params["timeHorizon"],
                                       self.params["timeHorizonObst"],
                                       self.robot.radius,
                                       self.max_speed)

        # robot
        robot = self.sim.addAgent(self.robot.position,
                                  self.params['neighborDist'],
                                  self.params['maxNeighbors'],
                                  self.params['timeHorizon'],
                                  self.params['timeHorizonObst'],
                                  self.robot.radius,
                                  self.robot.v_pref,
                                  self.robot.velocity)

        pref_velocity = self.robot.goal[0] - self.robot.position[0], \
                        self.robot.goal[1] - self.robot.position[1]

        if np.linalg.norm(pref_velocity) > 1:
            pref_velocity /= np.linalg.norm(pref_velocity)
        self.sim.setAgentPrefVelocity(robot, tuple(pref_velocity))

        # obstacles
        for i, dy_obstacle in enumerate(self.dy_obstacles_list):
            agent = self.sim.addAgent(dy_obstacle.position,
                                      self.params['neighborDist'],
                                      self.params['maxNeighbors'],
                                      self.params['timeHorizon'],
                                      self.params['timeHorizonObst'],
                                      dy_obstacle.radius,
                                      dy_obstacle.v_pref,
                                      dy_obstacle.velocity)

            pref_velocity = dy_obstacle.goal[0] - dy_obstacle.position[0], \
                            dy_obstacle.goal[1] - dy_obstacle.position[1]
            if np.linalg.norm(pref_velocity) > 1:
                pref_velocity /= np.linalg.norm(pref_velocity)
            self.sim.setAgentPrefVelocity(agent, tuple(pref_velocity))

    def data_collection(self, train_policy=True):
        # cnt episodes for set num of pretrain
        self.episode_cnt += 1

        # init robot reach
        robot_reached = False

        # logged robot and dy-obj trajectories
        while not robot_reached:
            # check robot collision
            robot_collision = False

            # change obstacles position (=reset)
            self.pretrain_()

            check_dy_obstacles_reach_goal = [0] * len(self.robot_with_dy_obstacles_list)  # rvo2의 목적지 도달 확인용
            check_reach_goal_pose = [0] * len(self.robot_with_dy_obstacles_list)  # rvo2의 목적지 도달 위치 기록용

            #  loop until logged trajectory of robot reached goal
            self.robot_n_dy_obstacles_states = [[] for _ in range(len(self.robot_with_dy_obstacles_list))]

            for step in range(self.max_steps):
                self.sim.doStep()

                if robot_reached:
                    break

                if robot_collision:
                    break

                # check collision
                robot_position = self.sim.getAgentPosition(0)
                for i, agent in enumerate(self.robot_with_dy_obstacles_list[1:], start=1):
                    ob_position = self.sim.getAgentPosition(i)
                    dis = np.array([(robot_position[0] - ob_position[0]),(robot_position[1] - ob_position[1])])
                    collision_distance_check = np.linalg.norm(dis)
                    if collision_distance_check < agent.radius + self.robot.radius:
                        robot_collision = True
                        break

                for i, agent in enumerate(self.robot_with_dy_obstacles_list):
                    # 목적지에 도달하면 멈추어서 정적 장애물 역할을 함
                    if not check_dy_obstacles_reach_goal[i]:
                        # 목적지 도달 체크
                        rvo2_dy_obstacle_pose = self.sim.getAgentPosition(i)
                        rvo2_dy_obstacle_velocity = self.sim.getAgentVelocity(i)
                        dy_obstacle_goal = agent.goal
                        reach_goal = np.linalg.norm(
                            np.array(rvo2_dy_obstacle_pose) - np.array(dy_obstacle_goal)) \
                                     < agent.radius + self.robot.goal_offset   # offset

                        if i == 0:  # robot state : px py vx vy gx gy r
                            if self.is_relative:
                                robot_goal_vec = (agent.goal[0] - rvo2_dy_obstacle_pose[0],
                                                  agent.goal[1] - rvo2_dy_obstacle_pose[1])
                            else:
                                robot_goal_vec = (agent.goal[0], agent.goal[1])

                            self.robot_n_dy_obstacles_states[i].append(
                                rvo2_dy_obstacle_pose + rvo2_dy_obstacle_velocity + robot_goal_vec + (agent.radius,))
                        else:       # obstacle state : px py vx vy r
                            self.robot_n_dy_obstacles_states[i].append(rvo2_dy_obstacle_pose + rvo2_dy_obstacle_velocity + (agent.radius,))

                        if reach_goal:
                            check_dy_obstacles_reach_goal[i] = reach_goal
                            check_reach_goal_pose[i] = rvo2_dy_obstacle_pose

                            if i == 0:
                                robot_reached = True

                    # 목적지 도달하면 그 자리에 멈춤
                    else:
                        self.sim.setAgentVelocity(i, (0, 0))
                        self.robot_n_dy_obstacles_states[i].append(check_reach_goal_pose[i] + (0, 0) + (agent.radius, ))

        # obstacles state : [px, py, vx, vy, radius]
        obstacles_states = self.robot_n_dy_obstacles_states

        # slicing obstacles state according to number of obstacles
        obstacles_states = np.array(obstacles_states, dtype=object)
        sliced_agents_states = np.array([obstacles_states[:,i] for i in range(len(obstacles_states[0]))])

        # transform log data for offline learning
        for i, ob in enumerate(sliced_agents_states[1:-1]):
            robot_ob = ob[0]
            # 위치, 속도를 로봇 기준 상대 좌표로 변환
            if self.is_relative:
                dy_obstacle_ob = [(dy_obstacle[0] - robot_ob[0], dy_obstacle[1] - robot_ob[1],
                                   dy_obstacle[2] - robot_ob[2], dy_obstacle[3] - robot_ob[3],
                                   dy_obstacle[4]) for dy_obstacle in ob[1:]]
            else:
                dy_obstacle_ob = [(dy_obstacle[0], dy_obstacle[1],
                                   dy_obstacle[2], dy_obstacle[3],
                                   dy_obstacle[4]) for dy_obstacle in ob[1:]]

            # 거리순으로 정렬 (먼 순으로)
            if self.is_obstacle_sort:
                dy_obstacle_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5, reverse=True)

            # state
            state = [robot_ob] + dy_obstacle_ob

            # reward
            reward, is_terminal = self.reward_function(state)

            # new_state
            new_ob = sliced_agents_states[i + 1]
            new_robot_ob = new_ob[0]
            if self.is_relative:
                new_dy_obstacle_ob = [(dy_obstacle[0] - robot_ob[0], dy_obstacle[1] - robot_ob[1],
                                       dy_obstacle[2] - robot_ob[2], dy_obstacle[3] - robot_ob[3],
                                       dy_obstacle[4]) for dy_obstacle in new_ob[1:]]
            else:
                new_dy_obstacle_ob = [(dy_obstacle[0], dy_obstacle[1],
                                       dy_obstacle[2], dy_obstacle[3],
                                       dy_obstacle[4]) for dy_obstacle in new_ob[1:]]

            # 거리순으로 정렬 (먼 순으로)
            if self.is_obstacle_sort:
                new_dy_obstacle_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5, reverse=True)

            new_state = [new_robot_ob] + new_dy_obstacle_ob

            # action
            action = new_robot_ob[2:4]
            action = np.array(action)
            action += np.random.normal(0.0, 1 * 0.1, size=2)
            action = action.clip(-1, 1)

            # flatten state data
            state = np.array(state_flattening(state))
            new_state = np.array(state_flattening(new_state))

            self.pretraied_replay_buffer.append((state, action, reward, new_state, is_terminal))

    def pretraining(self, pretrain_episodes=5000):
        # set transitions to replay buffer
        for (state, action, reward, new_state, is_terminal) in tqdm(self.pretraied_replay_buffer, desc="Transform State For Pretraining"):
            self.env.robot.policy.store_trajectory(state, action, reward, new_state, is_terminal)

        # pretrain policy
        for i in tqdm(range(pretrain_episodes), desc="Offline Learning Pretraining"):
            self.env.robot.policy.train()

    def render(self, obstacles_states):
        robot_states = obstacles_states[0]
        dy_obstacles_states = obstacles_states[1:]

        # color setting
        robot_color = 'chartreuse'
        static_obstacle_color = 'yellow'
        dynamic_obstacle_color = 'aqua'
        goal_color = 'red'

        fig, ax = plt.subplots(figsize=(self.env.square_width, self.env.square_height))
        ax.set_xlim(-int(self.env.square_width / 2), int(self.env.square_width / 2))
        ax.set_ylim(-int(self.env.square_width / 2), int(self.env.square_width / 2))
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # 에피소드, 스텝 시간 표시
        step_cnt = plt.text(self.env.square_width / 2, self.env.square_height / 2, 'Step : {}'.format(0), fontsize=16)
        ax.add_artist(step_cnt)

        # 목적지 그리기
        goal_x, goal_y = self.robot.goal
        goal = mlines.Line2D([goal_x], [goal_y], color=goal_color, marker='*', linestyle='None', markersize=15,
                             label='Goal')
        ax.add_artist(goal)
        # 도착지 허용범위 그리기
        goal_offset = Circle((goal_x, goal_y), self.robot.goal_offset, fill=True, color='salmon')
        ax.add_artist(goal_offset)

        # j개의 동적 장애물 이동 경로 그리기
        dy_obstacle_circle_list = []
        for j, dy_obstacle in enumerate(dy_obstacles_states):
            for k, dy_obstacle_state in enumerate(dy_obstacle):
                if k % 2 == 0:
                    dy_obstacle_circle = Circle(dy_obstacle_state[:2], dy_obstacle_state[4], fill=True, color=dynamic_obstacle_color, ec='black')
                    plt.text(dy_obstacle_state[0], dy_obstacle_state[1], "{}".format(k), fontsize=13, ha='center', fontweight='bold')
                    ax.add_artist(dy_obstacle_circle)

        for i, robot_state in enumerate(robot_states):
            if i % 2 == 0:
                robot_circle = Circle(robot_state[:2], robot_state[6], fill=True, color=robot_color, ec='black')
                plt.text(robot_state[0], robot_state[1], "{}".format(i), fontsize=10, ha='center', fontweight='bold')
                ax.add_artist(robot_circle)

        plt.show()





