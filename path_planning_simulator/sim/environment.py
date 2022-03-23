import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
import matplotlib.lines as mlines
import matplotlib.animation as animation
import rvo2
import gym
import time
from path_planning_simulator.utils.utils import *


class Environment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_rvo2=True):

        """
        환경 맵 세팅(main)
        장애물 배치(main) -> 세팅(Env) : 완료
            동적 장애물은 원
            정적 장애물은 사각형
        로봇 배치(main) -> 세팅(Env) : 완료

        run sim -> loop 돌면서 스텝 진행

        gym wrapping
        """

        # map 크기
        self.square_width = 10
        self.square_height = 10

        # 로봇과 장애물 소환
        self.robot = None
        self.dy_obstacles = []
        self.st_obstacles = []

        # reset() 할 때 초기 위치 설정
        self.init_position = None
        self.init_velocity = None
        self.init_goal = None

        # 한 에피소드당 행동 정보 기록용. reset()하면 내용 삭제됨
        self.dy_obstacles_positions = []
        self.dy_obstacles_velocities = []
        self.robot_position = None

        # 정답용 로봇 정보 기록용. reset()하면 초기화 됨
        self.answer_robot_pos_n_vel = []
        self.answer_robot_reach_goal = False

        # 한 에피소드당 스텝 시간 측정용
        self.time_step = None  # RVO2 의 step 시간 단위
        self.global_time = None
        self.time_limit = None

        self.step_cnt = 0

        # 충돌 거리 설정
        self.safe_distance = 1.0  # 로봇과 장애물이 소환되는 위치의 거리를 세팅

        # 상태 정보 설계용
        self.scailing_factor = 1

        # 보상함수 설계용
        self.target_norm = None

        # rvo2 실행
        self.start_rvo2 = start_rvo2
        if self.start_rvo2:
            self.params = {"neighborDist": 10, "maxNeighbors": 20, "timeHorizon": 5, "timeHorizonObst": 5}
            self.sim = None

    def set_time_step_and_time_limit(self, time_step, time_limit):
        self.time_step = time_step
        self.time_limit = time_limit

    def set_robot(self, robot):
        self.robot = robot
        self.init_position = robot.position
        self.init_velocity = robot.velocity
        self.init_goal = robot.goal
        self.robot_position = []

    def set_dynamic_obstacle(self, obstacle):
        # 장애물 수만큼 불러주어야 함
        self.dy_obstacles.append(obstacle)
        # 각 장애물 마다 위치 기록을 저장하므로 2차원 리스트 적용
        self.dy_obstacles_positions.append([])
        self.dy_obstacles_velocities.append([])

    def set_static_obstacle(self, obstacle):
        # 장애물 수만큼 불러주어야 함
        self.st_obstacles.append(obstacle)

    @property
    def dy_obstacles_list(self):
        return self.dy_obstacles

    @property
    def st_obstacles_list(self):
        return self.st_obstacles

    def step(self, robot_action):
        # action and update position
        self.robot.step(robot_action)
        # 로봇 위치 정보 저장
        self.robot_position.append(self.robot.position)

        if self.start_rvo2:
            # rvo2를 이용하여서 reset 단계에서 이미 장애물들의 이동 정보들을 다 얻은 경우
            # self.dy_obstacles_positions
            pass
        else:
            # 직접 장애물마다 정책을 장착하여 행동하는 경우
            # vx, vy 정보를 이용하여 각 step 적용 용도로 사용
            dy_obstacles_actions = []

            for dy_obstacle in self.dy_obstacles:
                # obstacles data without itself
                ob = [other_dy_obstacle.self_state_wo_goal for other_dy_obstacle in self.dy_obstacles if
                      other_dy_obstacle != dy_obstacle]  # tuple (px, py, vx, vy, radius)
                obstacle_action = dy_obstacle.act(ob)  # vx, vy
                dy_obstacles_actions.append(obstacle_action)

            for i, obstacle_action in enumerate(dy_obstacles_actions):
                # 각 장애물 행동 하고
                self.dy_obstacles[i].step(obstacle_action)
                # 장애물의 위치 정보 저장
                self.dy_obstacles_positions[i].append(self.dy_obstacles[i].position)

        self.global_time += 1
        self.step_cnt += 1

        # collision check btw robot and dynamic obstacle
        '''
        장애물 사이의 충돌은 고려하지 않음
        '''
        collision = False
        reach_goal = False

        # 동적 장애물 충돌 검사
        closest_dist_of_dy_obs = float("inf")
        for i, dy_obstacle in enumerate(self.dy_obstacles):
            if self.start_rvo2:
                dy_obstacle.px, dy_obstacle.py = self.dy_obstacles_positions[i][self.step_cnt]
                dy_obstacle.vx, dy_obstacle.vy = self.dy_obstacles_velocities[i][self.step_cnt]
            dx = dy_obstacle.px - self.robot.px
            dy = dy_obstacle.py - self.robot.py

            l2 = pow(dx, 2) + pow(dy, 2)
            l2_norm = pow(l2, 0.5)

            if closest_dist_of_dy_obs > l2_norm:
                closest_dist_of_dy_obs = l2_norm

            if closest_dist_of_dy_obs - self.robot.radius - dy_obstacle.radius < 0:
                # print("collision distance : ", closest_dist_of_dy_obs - self.robot.radius - dy_obstacle.radius)
                collision = True
                break

        # 정적 장애물 충돌 검사
        closest_dist_of_st_obs = float("inf")
        for i, st_obstacle in enumerate(self.st_obstacles):
            # 사각형 장애물에 대한 충돌 검사
            if st_obstacle.rectangle:
                # 사각형의 좌상단, 우하단의 점을 기준으로 현재 로봇의 위치에 대한 최단 거리의 점을 클램핑으로 계산
                rect_left_floor = (st_obstacle.px - (st_obstacle.width / 2), st_obstacle.py + (st_obstacle.height / 2))
                rect_right_bottom = (
                    st_obstacle.px + (st_obstacle.width / 2), st_obstacle.py - (st_obstacle.height / 2))

                clamped_x = max(rect_left_floor[0], min(rect_right_bottom[0], self.robot.px))
                clamped_y = max(rect_right_bottom[1], min(rect_left_floor[1], self.robot.py))

                # 투영 점 (= 로봇과 사각형 사이의 최단 거리의 점) 과 로봇의 위치에 대한 최단 거리 계산
                dx = self.robot.px - clamped_x
                dy = self.robot.py - clamped_y

                l2 = pow(dx, 2) + pow(dy, 2)
                l2_norm = pow(l2, 0.5)

                # 최단 거리가 원의 반지름보다 크면 충돌하지 않고 원의 반지름보다 작으면 충돌
                if l2_norm < self.robot.radius:
                    collision = True
                    break

            else:
                # 원 장애물에 대한 충돌 검사
                dx = st_obstacle.px - self.robot.px
                dy = st_obstacle.py - self.robot.py

                l2 = pow(dx, 2) + pow(dy, 2)
                l2_norm = pow(l2, 0.5)

                if closest_dist_of_st_obs > l2_norm:
                    closest_dist_of_st_obs = l2_norm

                if closest_dist_of_st_obs - self.robot.radius - st_obstacle.radius < 0:
                    # print("collision distance : ", closest_dist_of_st_obs - self.robot.radius - st_obstacle.radius)
                    collision = True
                    break

        # check reaching the goal
        reach_goal = self.robot.reach_goal()
        out_of_boundary = -self.square_width // 2 > self.robot.position[0] or \
                          self.square_width // 2 < self.robot.position[0] or \
                          -self.square_height // 2 > self.robot.position[1] or \
                          self.square_height // 2 < self.robot.position[1]

        # reward setting
        '''
        조건 : time, collision, reach_goal 
        '''
        # 0. time reward
        reward = -0.05

        # 1. reward for distance
        target_distance_vector = (
            self.robot.position[0] - self.robot.goal[0], self.robot.position[1] - self.robot.goal[1])
        target_norm = np.linalg.norm(target_distance_vector)

        if self.target_norm is None:
            self.target_norm = target_norm

        delta_reward = lambda x: np.tanh(x) if x > 0 else np.tanh(0.9 * x)

        reward += delta_reward(self.target_norm - target_norm)

        self.target_norm = target_norm

        # 2. reward for terminal
        if reach_goal:
            reward += 10
            done = True
            info = "Goal"
            # print("goal!")
            self.target_norm = None

        elif collision:
            reward = -5
            done = True
            info = "Collision"
            # print("collision!")
            self.target_norm = None

        elif self.global_time >= self.time_limit - 1:
            reward = -5
            done = True
            info = "TimeOut"
            # print("timeout!")
            self.target_norm = None

        # out of map get negative reward
        elif out_of_boundary:
            reward = -5
            done = True
            info = "OutBoundary"
            # print("collision!")
            self.target_norm = None

        else:
            reward += 0
            done = False
            info = None

        ####Assistant Model####
        # 목적지에 도착하지 못하였을 때, 정답 로봇의 경로를 이용하여 trajectories를 생성하고 replay buffer 에 넣어준다. (성능향상을 확인하지 못함)
        # if (collision or (self.global_time >= self.time_limit - 1) or out_of_boundary) and self.answer_robot_reach_goal:
        #     # reset() 에서 저장한 정답 로봇 정보와 장애물 데이터를 이용하여 trajectories 생성한다.
        #     print("Assistant work")
        #     self.set_answer_trajectories_to_buffer()

        # next_step_ob
        next_robot_ob = [robot_state_data for robot_state_data in self.robot.self_state_w_goal]
        # next_dy_obstacle_ob = [dy_obstacle.self_state_wo_goal for dy_obstacle in self.dy_obstacles]
        # 로봇 관점에서 로봇의 감지 범위 내에 있는 장애물들의 방향 벡터를 장애물의 위치 정보대신 사용한다.
        next_dy_obstacle_ob = [
                                (dy_obstacle.px - self.robot.px, dy_obstacle.py - self.robot.py, dy_obstacle.vx - self.robot.vx,
                                 dy_obstacle.vy - self.robot.vy, dy_obstacle.radius) for dy_obstacle in self.dy_obstacles if
                                pow(pow((dy_obstacle.px - self.robot.px), 2) + pow((dy_obstacle.py - self.robot.py), 2),
                                    0.5) <= self.robot.detection_scope]
        next_st_obstacle_ob = [st_obstacle.self_state_wo_goal_rectangle for st_obstacle in self.st_obstacles]

        # 거리순으로 정렬 (먼 순으로)
        next_dy_obstacle_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5,
                                 reverse=True)

        # 회전변환하여 로봇에 대한 상대좌표로 변환
        # if 상대좌표:
            # next_robot_ob = [(next_robot_ob[0], next_robot_ob[1],
            #                   *rotate2D([next_robot_ob[2], next_robot_ob[3]], (np.pi / 2 - self.robot.theta)),
            #                   next_robot_ob[4], next_robot_ob[5], next_robot_ob[6])]

            # next_dy_obstacle_ob = [(*rotate2D([dy_obstacle[0], dy_obstacle[1]], (np.pi / 2 - self.robot.theta)),
            #                         *rotate2D([dy_obstacle[2], dy_obstacle[3]], (np.pi / 2 - self.robot.theta)),
            #                         dy_obstacle[4]) for dy_obstacle in next_dy_obstacle_ob]

        next_ob = [next_robot_ob] + [next_dy_obstacle_ob] + [next_st_obstacle_ob]

        next_state = list(itertools.chain(*next_ob))
        next_state_flatten = []  # 내부 튜플 제거...
        for item in next_state:
            if isinstance(item, tuple):
                if len(item) != 0:
                    for x in item:
                        next_state_flatten.append(x)
                else:
                    pass
            else:
                next_state_flatten.append(item)

        next_state = next_state_flatten

        return np.array(next_state) / self.scailing_factor, reward, done, info

    def reset(self, random_position=False, random_goal=False, max_steps=1000):
        # 에피소드 실행 시간 초기화
        self.global_time = 0
        self.step_cnt = 0

        # 로봇, 장애물의 위치 정보 초기화
        self.robot_position = []
        self.dy_obstacles_positions = [[] for _ in range(len(self.dy_obstacles_list))]
        self.dy_obstacles_velocities = [[] for _ in range(len(self.dy_obstacles_list))]

        # 정답용 로봇 정보 초기화
        self.answer_robot_pos_n_vel = []
        self.answer_robot_reach_goal = False

        # 로봇 위치, 속도, 목적지 초기화
        # random 적용
        sign = 1 if np.random.random() > 0.5 else -1
        if random_position:
            robot_px = np.random.random() * (self.square_width * 0.5) * sign
            robot_py = (np.random.random() - 0.5) * self.square_height
            self.robot.px = robot_px
            self.robot.py = robot_py
            self.robot.vx = 0
            self.robot.vy = 0
        else:
            self.robot.px, self.robot.py = self.init_position
            self.robot.vx, self.robot.vy = self.init_velocity  # if non-holonomic : vx = angular velocity, vx = linear velocity
            self.robot.theta = np.pi / 2  # world y axis direction is init direction of robot

        if random_goal:
            robot_gx = np.random.random() * (self.square_width * 0.5) * sign
            robot_gy = (np.random.random() - 0.5) * self.square_height
            self.robot.gx = robot_gx
            self.robot.gy = robot_gy
        else:
            self.robot.gx, self.robot.gy = self.init_goal

        # 장애물 위치 세팅
        self.generate_random_position()

        if self.start_rvo2:
            # self.dy_obstacles_positions 을 rvo2를 이용하여 정보를 얻는다.
            radius = 0.3
            max_speed = 1
            self.sim = rvo2.PyRVOSimulator(self.time_step, self.params['neighborDist'], self.params['maxNeighbors'],
                                           self.params['timeHorizon'], self.params['timeHorizonObst'], radius,
                                           max_speed)

            for i, dy_obstacle in enumerate(self.dy_obstacles_list):
                # 장애물 정보 추가
                agent = self.sim.addAgent(dy_obstacle.position, self.params['neighborDist'],
                                          self.params['maxNeighbors'],
                                          self.params['timeHorizon'], self.params['timeHorizonObst'],
                                          dy_obstacle.radius,
                                          dy_obstacle.v_pref, dy_obstacle.velocity)

                pref_velocity = dy_obstacle.goal[0] - dy_obstacle.position[0], \
                                dy_obstacle.goal[1] - dy_obstacle.position[1]
                if np.linalg.norm(pref_velocity) > 1:
                    pref_velocity /= np.linalg.norm(pref_velocity)
                self.sim.setAgentPrefVelocity(agent, tuple(pref_velocity))

            # 정답용 로봇 정보를 추가 (Offline Learning 에 사용)
            answer_robot = self.sim.addAgent(self.robot.position,
                                             self.params['neighborDist'],
                                             self.params['maxNeighbors'],
                                             self.params['timeHorizon'],
                                             self.params['timeHorizonObst'],
                                             self.robot.radius,
                                             self.robot.v_pref,
                                             self.robot.velocity)

            answer_robot_pref_velocity = self.robot.goal[0] - self.robot.position[0], \
                                         self.robot.goal[1] - self.robot.position[1]
            if np.linalg.norm(answer_robot_pref_velocity) > 1:
                answer_robot_pref_velocity /= np.linalg.norm(answer_robot_pref_velocity)
            self.sim.setAgentPrefVelocity(answer_robot, tuple(answer_robot_pref_velocity))

            # print('Simulation has %i agents and %i obstacle vertices in it.' %
            #       (self.sim.getNumAgents(), self.sim.getNumObstacleVertices()))

            check_dy_obstacles_reach_goal = [0] * len(self.dy_obstacles_list)  # rvo2의 목적지 도달 확인용
            check_reach_goal_pose = [0] * len(self.dy_obstacles_list)  # rvo2의 목적지 도달 위치 기록용
            # 장애물의 주행 경로 정보를 저장 + 로봇이 RVO2 알고리즘을 주행하는 경우의 경로 추가 저장
            for step in range(max_steps):
                self.sim.doStep()

                for i, dy_obstacle in enumerate(self.dy_obstacles_list):
                    # 목적지에 도달하면 멈추어서 정적 장애물 역할을 함
                    if not check_dy_obstacles_reach_goal[i]:
                        # 목적지 도달 체크
                        rvo2_dy_obstacle_pose = self.sim.getAgentPosition(i)
                        rvo2_dy_obstacle_velocity = self.sim.getAgentVelocity(i)
                        dy_obstacle_goal = dy_obstacle.goal
                        reach_goal = np.linalg.norm(
                            np.array(rvo2_dy_obstacle_pose) - np.array(dy_obstacle_goal)) < dy_obstacle.radius
                        out_of_boundary = rvo2_dy_obstacle_pose[0] < -self.square_width // 2 or self.square_width // 2 > \
                                          rvo2_dy_obstacle_pose[0] or -self.square_height // 2 > rvo2_dy_obstacle_pose[
                                              1] or self.square_height // 2 < rvo2_dy_obstacle_pose[1]

                        if reach_goal or out_of_boundary:
                            check_dy_obstacles_reach_goal[i] = reach_goal
                            check_reach_goal_pose[i] = rvo2_dy_obstacle_pose

                        self.dy_obstacles_positions[i].append(rvo2_dy_obstacle_pose)
                        self.dy_obstacles_velocities[i].append(rvo2_dy_obstacle_velocity)

                    # 목적지 도달하면 그 자리에 멈춤
                    else:
                        self.sim.setAgentVelocity(i, (0, 0))
                        self.dy_obstacles_positions[i].append(check_reach_goal_pose[i])
                        self.dy_obstacles_velocities[i].append((0, 0))

                # 정답용 로봇 경로 저장
                '''
                simulation에서 로봇이 충돌하였을 때 정답지의 도착 여부 확인 후 도착하였으면 정답 기록과 장애물 정보를 이용하여 trajectory
                구성하여 replay buffer에 저장
                step() 에서 목적지 도달 실패하면 그때 저장된 정답용 로봇 경로 데이터로 trajectory 생성
                '''
                if not self.answer_robot_reach_goal:
                    # 정답용 로봇 경로 저장
                    robot_idx = len(self.dy_obstacles_list)     # 장애물 세팅 후 로봇을 세팅하여서 마지막 위치가 로봇의 정보가 있다.
                    answer_robot_pose = self.sim.getAgentPosition(robot_idx)
                    answer_robot_velocity = self.sim.getAgentVelocity(robot_idx)
                    self.answer_robot_pos_n_vel.append((*answer_robot_pose, *answer_robot_velocity))

                    # 목적지 도달 검사
                    answer_robot_goal = self.robot.goal
                    self.answer_robot_reach_goal = np.linalg.norm(np.array(answer_robot_pose) - np.array(answer_robot_goal)) \
                                                    < self.robot.radius + self.robot.goal_offset

            # 정답용 로봇이 목적지 도달하지 못하였다면 해당 에피소드의 기록을 지운다.
            if not self.answer_robot_reach_goal:
                self.answer_robot_pos_n_vel.clear()

        # 로봇과 장애물의 상태 정보 출력
        # 로봇의 탐지 범위는 detection scope이다.
        robot_ob = [robot_state_data for robot_state_data in self.robot.self_state_w_goal]
        # dy_obstacle_ob = [dy_obstacle.self_state_wo_goal for dy_obstacle in self.dy_obstacles]
        # 로봇 관점에서 로봇의 감지 범위 내에 있는 장애물들의 방향 벡터를 장애물의 위치 정보대신 사용한다.
        dy_obstacle_ob = [(dy_obstacle.px - self.robot.px, dy_obstacle.py - self.robot.py,
                           dy_obstacle.vx - self.robot.vx, dy_obstacle.vy - self.robot.vy,
                           dy_obstacle.radius)
                          for dy_obstacle in self.dy_obstacles if
                          pow(pow((dy_obstacle.px - self.robot.px), 2) + pow((dy_obstacle.py - self.robot.py), 2),
                              0.5) <= self.robot.detection_scope]
        st_obstacle_ob = [st_obstacle.self_state_wo_goal_rectangle for st_obstacle in self.st_obstacles]

        # 멀리 있는 순으로 장애물을 정렬한다.
        dy_obstacle_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5, reverse=True)

        # 회전변환하여 로봇에 대한 상대좌표로 변환, holomonic 일때
        # if 상대좌표:
            # robot_ob = [(robot_ob[0], robot_ob[1],
            #             *rotate2D([robot_ob[2], robot_ob[3]], (np.pi / 2 - self.robot.theta)),
            #             robot_ob[4], robot_ob[5], robot_ob[6])]

            # dy_obstacle_ob = [(*rotate2D([dy_obstacle[0], dy_obstacle[1]], (np.pi / 2 - self.robot.theta)),
            #                    *rotate2D([dy_obstacle[2], dy_obstacle[3]], (np.pi / 2 - self.robot.theta)),
            #                    dy_obstacle[4]) for dy_obstacle in dy_obstacle_ob]

        ob = [robot_ob] + [dy_obstacle_ob] + [st_obstacle_ob]

        state = list(itertools.chain(*ob))
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

        state = state_flatten

        return np.array(state) / self.scailing_factor

    def set_answer_trajectories_to_buffer(self):
        total_steps = len(self.answer_robot_pos_n_vel)
        robot_goal = self.robot.goal
        robot_radius = self.robot.radius
        # 로봇이 목적지에 도달한 경로 정보까지만 저장해준다.
        for step_cnt in range(total_steps - 1):
            # answer state
            step_dy_obstacles_ob = []
            robot_pose = self.answer_robot_pos_n_vel[step_cnt][:2]
            robot_vel = self.answer_robot_pos_n_vel[step_cnt][2:]
            step_robot_ob = [self.answer_robot_pos_n_vel[step_cnt] + (robot_goal[0] - robot_pose[0], robot_goal[1] - robot_pose[1], robot_radius)]
            for i, dy_obstacle in enumerate(self.dy_obstacles):
                # 로봇의 센서 탐지 범위는 고려하지 않았다.
                dy_obstacle_px, dy_obstacle_py = self.dy_obstacles_positions[i][step_cnt]
                dy_obstacle_vx, dy_obstacle_vy = self.dy_obstacles_velocities[i][step_cnt]
                dy_obstacle_state = (dy_obstacle_px - robot_pose[0], dy_obstacle_py - robot_pose[1],
                                     dy_obstacle_vx - robot_vel[0], dy_obstacle_vy - robot_vel[1], dy_obstacle.radius)
                step_dy_obstacles_ob.append(dy_obstacle_state)

            # 멀리 있는 순으로 장애물을 정렬한다.
            step_dy_obstacles_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5, reverse=True)

            # holonomic은 차후에 고려

            # 정적 장애물 제외
            ob = step_robot_ob + step_dy_obstacles_ob

            step_state = list(itertools.chain(*ob))
            state_flatten = []  # 내부 튜플 제거...
            for item in step_state:
                if isinstance(item, tuple):
                    if len(item) != 0:
                        for x in item:
                            state_flatten.append(x)
                    else:
                        pass
                else:
                    state_flatten.append(item)
            step_state = state_flatten

            # 보상함수 처리
            reward, is_terminal = self.set_answer_reward_function(step_state)

            # answer new_state
            new_step_dy_obstacles_ob = []
            new_robot_pose = self.answer_robot_pos_n_vel[step_cnt+1][:2]
            new_robot_vel = self.answer_robot_pos_n_vel[step_cnt+1][2:]
            new_step_robot_ob = [self.answer_robot_pos_n_vel[step_cnt+1] +
                                 (robot_goal[0] - new_robot_pose[0], robot_goal[1] - new_robot_pose[1], robot_radius)]
            for j, new_dy_obstacle in enumerate(self.dy_obstacles):
                # 로봇의 센서 탐지 범위는 고려하지 않았다.
                new_dy_obstacle_px, new_dy_obstacle_py = self.dy_obstacles_positions[j][step_cnt+1]
                new_dy_obstacle_vx, new_dy_obstacle_vy = self.dy_obstacles_velocities[j][step_cnt+1]
                dy_obstacle_state = (new_dy_obstacle_px - new_robot_pose[0], new_dy_obstacle_py - new_robot_pose[1],
                                     new_dy_obstacle_vx - new_robot_vel[0], new_dy_obstacle_vy - new_robot_vel[1],
                                     new_dy_obstacle.radius)
                new_step_dy_obstacles_ob.append(dy_obstacle_state)

            # 멀리 있는 순으로 장애물을 정렬한다.
            new_step_dy_obstacles_ob.sort(key=lambda x: (pow(x[0], 2) + pow(x[1], 2)) ** 0.5, reverse=True)

            # holonomic은 차후에 고려

            # 정적 장애물 제외
            new_ob = new_step_robot_ob + new_step_dy_obstacles_ob

            new_step_state = list(itertools.chain(*new_ob))
            new_state_flatten = []  # 내부 튜플 제거...
            for item in new_step_state:
                if isinstance(item, tuple):
                    if len(item) != 0:
                        for x in item:
                            new_state_flatten.append(x)
                    else:
                        pass
                else:
                    new_state_flatten.append(item)
            new_step_state = new_state_flatten

            # action 추출
            action = np.array(new_robot_vel)
            action += np.random.normal(0.0, 1 * 0.1, size=2)
            action = action.clip(-1, 1)

            self.robot.policy.store_trajectory(step_state, action, reward, new_step_state, is_terminal)

    def set_answer_reward_function(self, state):
        # shape : (robot+obstacle Num, 5)
        # reward for robot position

        robot_position = state[:2]
        robot_velocity = state[2:4]

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

    def generate_random_position(self):
        '''
        로봇이 우선 배치되고 해당 배치된 위치를 기준으로 장애물 배치
        '''

        robot_px, robot_py = self.robot.position
        robot_gx, robot_gy = self.robot.goal
        no_collision_distance = self.safe_distance

        for i, dy_obstacle in enumerate(self.dy_obstacles):
            # obstacles 의 속성들(v_pref, radius) 변경
            dy_obstacle.v_pref = np.random.uniform(0.5, 1.5)
            dy_obstacle.radius = np.random.uniform(0.2, 0.3)

            # 장애물 위치 변경 및 목적지 설정
            # 로봇의 위치에서 일정 거리 떨어진 곳에서 랜덤 생성
            sign = 1 if np.random.random() > 0.5 else -1  # 장애물의 위치와 목적지 위치를 서로 반대 방향에 설정하기 위함
            sign2 = 1 if np.random.random() > 0.5 else -1  # 일정 확률로 대각선 방향으로 이동하도록
            while True:
                # 맵의 중심을 (0,0)으로 기준 잡았을 경우
                obstacle_px = np.random.random() * (self.square_width * 0.5) * sign
                obstacle_py = (np.random.random() - 0.5) * self.square_height

                # 로봇과 랜덤 배치된 장애물의 거리를 재고 안전거리 이상 떨어져 있으면 장애물 위치 설정
                if np.linalg.norm((robot_px - obstacle_px,
                                   robot_py - obstacle_py)) - dy_obstacle.radius - self.robot.radius > no_collision_distance:
                    dy_obstacle.px = obstacle_px
                    dy_obstacle.py = obstacle_py
                    break

            # 장애물의 목적지는 맵을 4등분 하였을때 소환위치의 다른 영역을 향하도록 설정
            while True:
                # 맵의 중심을 (0,0)으로 기준 잡았을 경우
                obstacle_gx = np.random.random() * (self.square_width * 0.5) * -sign
                obstacle_gy = (np.random.random() - 0.5) * self.square_height * sign2

                # 장애물의 목적지가 로봇의 목적지와 안전거리 이상 떨어져 있게 목적지 위치 설정
                if np.linalg.norm((robot_gx - obstacle_gx,
                                   robot_gy - obstacle_gy)) - dy_obstacle.radius - self.robot.radius > no_collision_distance:
                    dy_obstacle.gx = obstacle_gx
                    dy_obstacle.gy = obstacle_gy
                    break

    def generate_circle_position(self, circle_radius=4):
        robot_px, robot_py = self.robot.position
        robot_gx, robot_gy = self.robot.goal
        no_collision_distance = self.safe_distance

        for i, dy_obstacle in enumerate(self.dy_obstacles):
            # obstacles 의 속성들(v_pref, radius) 변경
            dy_obstacle.v_pref = np.random.uniform(0.5, 1.5)
            dy_obstacle.radius = np.random.uniform(0.2, 0.3)

            while True:
                angle = np.random.random() * (2 * np.pi)
                obstacle_px = circle_radius * np.cos(angle)
                obstacle_py = circle_radius * np.sin(angle)

                # 로봇과 랜덤 배치된 장애물의 거리를 재고 안전거리 이상 떨어져 있으면 장애물 위치 설정
                if np.linalg.norm((robot_px - obstacle_px,
                                   robot_py - obstacle_py)) - dy_obstacle.radius - self.robot.radius > no_collision_distance:
                    dy_obstacle.px = obstacle_px
                    dy_obstacle.py = obstacle_py
                    dy_obstacle.gx = -obstacle_px
                    dy_obstacle.gy = -obstacle_py
                    break

    def render(self, path_info=True, is_plot=False):
        # color setting
        robot_color = 'green'
        static_obstacle_color = 'yellow'
        dynamic_obstacle_color = 'blue'
        goal_color = 'red'

        fig, ax = plt.subplots(figsize=(self.square_width, self.square_height))
        ax.set_xlim(-int(self.square_width / 2), int(self.square_width / 2))
        ax.set_ylim(-int(self.square_width / 2), int(self.square_width / 2))
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # 에피소드, 스텝 시간 표시
        step_cnt = plt.text(self.square_width / 2, self.square_height / 2, 'Step : {}'.format(0), fontsize=16)
        ax.add_artist(step_cnt)

        # 초기 로봇 그리기
        robot_circle = Circle(self.robot_position[0], self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot_circle)

        # 로봇 탐지 범위 그리기
        # robot_detection_circle = Circle(self.robot_position[0], self.robot.detection_scope, fill=True, color='mintcream')
        # ax.add_artist(robot_detection_circle)

        # 목적지 그리기
        goal_x, goal_y = self.robot.goal
        goal = mlines.Line2D([goal_x], [goal_y], color=goal_color, marker='*', linestyle='None', markersize=15,
                             label='Goal')
        ax.add_artist(goal)
        # 도착지 허용범위 그리기
        goal_offset = Circle((goal_x, goal_y), self.robot.goal_offset, fill=True, color='salmon')
        ax.add_artist(goal_offset)

        # j개의 초기 동적 장애물 그리기
        dy_obstacle_circle_list = []
        for j, dy_obstacle in enumerate(self.dy_obstacles_list):
            j_th_dy_obstacle_positions = self.dy_obstacles_positions[j]
            dy_obstacle_circle = Circle(j_th_dy_obstacle_positions[0], dy_obstacle.radius, fill=True,
                                        color=dynamic_obstacle_color)
            dy_obstacle_circle_list.append(dy_obstacle_circle)
            ax.add_artist(dy_obstacle_circle)

            # 동적 장애물의 목적지와 목적지 경로 표시
            if path_info:
                goal_circle = Circle(dy_obstacle.goal, 0.05, fill=True, color='black')
                ax.add_artist(goal_circle)

                goal_direction_line = ConnectionPatch(j_th_dy_obstacle_positions[0], dy_obstacle.goal, "data", "data",
                                                      arrowstyle="-|>", shrinkA=5, shrinkB=5, mutation_scale=20, fc="w")
                ax.add_artist(goal_direction_line)

        # 정적 장애물 그리기
        for st_obstacle in self.st_obstacles_list:
            if st_obstacle.rectangle:
                # 사각형 정적 장애물
                x_for_rect = st_obstacle.px - (st_obstacle.width / 2)
                y_for_rect = st_obstacle.py - (st_obstacle.height / 2)
                st_obstacle_rectangle = Rectangle((x_for_rect, y_for_rect), st_obstacle.width, st_obstacle.height,
                                                  angle=0.0)
                ax.add_artist(st_obstacle_rectangle)
            else:
                # 원형 정적 장애물
                st_obstacle_circle = Circle(st_obstacle.position, st_obstacle.radius, fill=True,
                                            color=static_obstacle_color)
                ax.add_artist(st_obstacle_circle)

        dy_obstacles_positions = self.dy_obstacles_positions

        def animate(frame):
            if frame == len(self.robot_position) - 1:

                # print('steps done. closing!')
                # plt.ion()
                plt.close(fig)

            else:
                # 로봇의 위치 기록을 기반으로 움직이기
                robot_circle.center = self.robot_position[frame]
                # 동적 장애물 위치 기록을 기반으로 움직이기
                for k, dy_obst in enumerate(dy_obstacle_circle_list):
                    k_th_dy_obst_positions = dy_obstacles_positions[k]
                    dy_obst.center = k_th_dy_obst_positions[frame]

                step_cnt.set_text('Step : {}'.format(frame + 1))

        f = r"./learning_data/video/"
        timestr = time.strftime("%m%d%H%M")
        ani = animation.FuncAnimation(fig, animate, frames=len(self.robot_position), repeat=False)
        if is_plot:
            plt.show()
        ani.save(f + timestr + ".gif", writer='imagemagick', fps=30)
