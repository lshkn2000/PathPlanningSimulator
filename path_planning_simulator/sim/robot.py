import itertools
from collections import deque
from collections import namedtuple

import numpy as np

from path_planning_simulator.sim.agent import Agent


class Robot(Agent):
    def __init__(self, discrete_action_space=None, robot_name="Robot"):
        super(Robot, self).__init__()
        self.name = robot_name
        # self.Action = namedtuple('Action', ['vx', 'vy'])
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
            velocity = 1
            vx = velocity * np.cos(self.discrete_rad_angles[action])
            vy = velocity * np.sin(self.discrete_rad_angles[action])
            # self.Action.vx = vx
            # self.Action.vy = vy
            self.action[0] = vx
            self.action[1] = vy

            return self.action, action      # action은 discrete action 에 대한 Index

        # 2. Continuous라면 vx, vy 에 대한 연속적인 값이 나오므로 vx, vy를 바로 넣어주면 된다.
        elif isinstance(action, np.ndarray):
            # self.Action.vx = action[0]
            # self.Action.vy = action[1]
            self.action[0] = action[0]
            self.action[1] = action[1]
        else:
            print("action : ", action)
            print("action type : {}".format(type(action)))
            raise Exception("action의 형태를 확인하세요.")

            return self.action, None
