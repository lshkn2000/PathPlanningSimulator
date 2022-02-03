import collections
import itertools
import numpy as np
import torch
import rvo2


device = "cuda:0" if torch.cuda.is_available() else "cpu"


class PretrainedSim(object):
    def __init__(self, env, time_step, robot_with_dy_obstacles_list, max_step=200, max_speed=1):
        self.sim = None
        self.robot_position = None
        self.target_norm = None

        self.robot = robot_with_dy_obstacles_list[0]
        self.dy_obstacles_list = robot_with_dy_obstacles_list[1:]
        self.robot_with_dy_obstacles_list = robot_with_dy_obstacles_list

        self.max_steps = max_step
        self.max_speed = max_speed
        self.env = env
        self.time_step = time_step
        self.params = {"neighborDist": 10, "maxNeighbors": 20, "timeHorizon": 5, "timeHorizonObst": 5}

        self.pretraied_replay_buffer = collections.deque(maxlen=100000)

    def reset(self):
        self.env.reset()
        self.dy_obstacles_list = self.env.dy_obstacles_list
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

        delta_reward = lambda x: 5 * np.tanh(x) if x > 0 else np.tanh(0.9 * x)

        reward += delta_reward(self.target_norm - target_norm)

        self.target_norm = target_norm

        reach_goal = np.linalg.norm(
            robot_position - np.array(self.robot.goal)) < robot_radius + 0.1  # offset

        # 2. reward for terminal
        if reach_goal:
            reward += 10
            done = True
            info = "Goal"
            self.target_norm = None

        else:
            reward += 0
            done = False
            info = None

        return reward, done

    def flatten_(self, state):
        state = list(itertools.chain(*state))
        state_flatten = []  # 내부 튜플 제거...
        for item in state:
            if isinstance(item, tuple):
                if len(item) != 0:
                    for x in item:
                        state_flatten.append(x)
                else:
                    pass
            else:
                state_flatten.append(item)

        return state_flatten

    def pretrain_(self):
        self.reset()

        obstacle_num = len(self.dy_obstacles_list)

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

    def pretrain(self):
        check_dy_obstacles_reach_goal = [0] * len(self.robot_with_dy_obstacles_list)  # rvo2의 목적지 도달 확인용
        check_reach_goal_pose = [0] * len(self.robot_with_dy_obstacles_list)  # rvo2의 목적지 도달 위치 기록용
        robot_reached = False

        # logged robot and dy-obj trajectories
        while not robot_reached:
            # change obstacles position (=reset)
            self.pretrain_()

            #  loop until logged trajectory of robot reached goal
            self.dy_obstacles_states = [[] for _ in range(len(self.robot_with_dy_obstacles_list))]

            for step in range(self.max_steps):
                self.sim.doStep()

                if robot_reached:
                    break

                for i, agent in enumerate(self.robot_with_dy_obstacles_list):
                    # 목적지에 도달하면 멈추어서 정적 장애물 역할을 함
                    if not check_dy_obstacles_reach_goal[i]:
                        # 목적지 도달 체크
                        rvo2_dy_obstacle_pose = self.sim.getAgentPosition(i)
                        rvo2_dy_obstacle_velocity = self.sim.getAgentVelocity(i)
                        dy_obstacle_goal = agent.goal
                        reach_goal = np.linalg.norm(
                            np.array(rvo2_dy_obstacle_pose) - np.array(dy_obstacle_goal)) < agent.radius + 0.1 # offset

                        if reach_goal:
                            check_dy_obstacles_reach_goal[i] = reach_goal
                            check_reach_goal_pose[i] = rvo2_dy_obstacle_pose

                            if i == 0:
                                robot_reached = True

                        if i == 0:  # robot state : px py vx vy gx gy r
                            self.dy_obstacles_states[i].append(
                                rvo2_dy_obstacle_pose + rvo2_dy_obstacle_velocity + agent.goal + (agent.radius,))
                        else:       # obstacle state : px py vx vy r
                            self.dy_obstacles_states[i].append(rvo2_dy_obstacle_pose + rvo2_dy_obstacle_velocity + (agent.radius,))

                    # 목적지 도달하면 그 자리에 멈춤
                    else:
                        self.sim.setAgentVelocity(i, (0, 0))
                        self.dy_obstacles_states[i].append(check_reach_goal_pose[i] + (0, 0) + (agent.radius, ))

        # obstacles state : [px, py, vx, vy, radius]
        obstacles_states = self.dy_obstacles_states

        # slicing obstacles state according to number of obstacles
        obstacles_states = np.array(obstacles_states, dtype=object)
        sliced_agents_states = np.array([obstacles_states[:,i] for i in range(len(obstacles_states[0]))])

        for i, state in enumerate(sliced_agents_states[:-1]):
            reward, is_terminal = self.reward_function(state)
            new_state = sliced_agents_states[i+1]
            action = sliced_agents_states[i+1][0][2:4]

            # flatten state data
            state = self.flatten_(state)
            new_state = self.flatten_(new_state)

            self.env.robot.policy.store_trajectory(state, action, reward, new_state, is_terminal)

            self.pretraied_replay_buffer.append((state, action, reward, new_state, is_terminal))

        if len(self.env.robot.policy.replay_buffer) > self.env.robot.policy.replay_buffer.batch_size:
            self.env.robot.policy.train()
            self.env.robot.policy.update_network()

    def save_model(self):
        self.env.robot.policy.save("learning_data/tmp")

    def get_pretrained_replay_buffer(self):
        return self.pretraied_replay_buffer



