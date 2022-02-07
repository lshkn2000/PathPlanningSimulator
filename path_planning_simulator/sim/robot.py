import itertools
from collections import deque
from collections import namedtuple

import numpy as np

from path_planning_simulator.sim.agent import Agent


class Robot(Agent):
    def __init__(self, discrete_action_space=None, is_holomonic=False, robot_name="Robot"):
        super(Robot, self).__init__()
        self.name = robot_name
        self.is_holonomic = is_holomonic
        self.theta = np.pi / 2 # 로봇이 북쪽 방향을 바라보는 것을 초기 방향으로 설정
        self.action = deque([None, None], maxlen=2)
        self.is_discrete_actions = False
        if discrete_action_space is not None:
            self.discrete_rad_angles = [(0 + (2 * np.pi / discrete_action_space) * i) for i in range(1, discrete_action_space+1)]
            self.is_discrete_actions = True

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")

        # 로봇이 받는 상태정보는 3가지의 리스트로 구성되어있다
        # 0. 로봇의 상태정보 [px, py, vx, vy, gx, gy, radius]
        # 1. 동적 장애물들의 상태정보 [(px, py, vx, vy, radius), ...]
        # 2. 정적 장애물의 상태정보 [(px, py, width, height)]

        # set state information
        # ob의 리스트 차원 줄이기
        state = ob

        # choose action using state by policy
        action = self.policy.predict(state)

        # action 이 discrete 인가 continuous인가
        # 1. dicrete이라면 0~num(acttion space) 만큼의 인덱스가 출력되므로 이를 discrete 에 맏게 변환해준다.
        if (isinstance(action, int) or isinstance(action, np.int64)) and self.is_discrete_actions:
            velocity = 2
            vx = velocity * np.cos(self.discrete_rad_angles[action])
            vy = velocity * np.sin(self.discrete_rad_angles[action])
            self.action[0] = vx
            self.action[1] = vy
            return self.action, action      # action은 discrete action 에 대한 Index

        # 2. Continuous라면 angle_velocity, linear_velocity 에 대한 연속적인 값이 나오므로 vx, vy를 바로 넣어주면 된다.
        elif isinstance(action, np.ndarray):
            if self.is_holonomic:
                self.action[0] = action[0]
                self.action[1] = action[1]
            else: # action[0] 는 각속도 action[1]은 선속도
                self.theta = self.theta + (action[0] * np.pi)   # rad/s
                # scope angle to -2pi ~ 2pi
                rot_delta_theta = self.theta / (2 * np.pi)
                rot_delta_theta = (rot_delta_theta - np.trunc(rot_delta_theta)) * (2 * np.pi)

                self.action[0] = action[1] * np.cos(rot_delta_theta)
                self.action[1] = action[1] * np.sin(rot_delta_theta)
            return self.action, None

        else:
            print("action : ", action)
            print("action type : {}".format(type(action)))
            raise Exception("action의 형태를 확인하세요. 넘파이 이어야 합니다.")

