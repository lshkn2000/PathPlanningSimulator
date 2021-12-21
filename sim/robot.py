import itertools
from collections import namedtuple
from sim.agent import Agent


Action = namedtuple('Action', ['vx', 'vy'])


class Robot(Agent):
    def __init__(self, robot_name="Robot"):
        super(Robot, self).__init__()
        self.name = robot_name


    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")

        # 로봇이 받는 상태정보는 3가지의 리스트로 구성되어있다
        # 0. 로봇의 상태정보 [px, py, vx, vy, gx, gy, radius]
        # 1. 동적 장애물들의 상태정보 [(px, py, vx, vy, radius), ...]
        # 2. 정적 장애물의 상태정보 [(px, py, width, height)]

        # set state information
        # ob의 리스트 차원 줄이기
        # state = ob
        state = list(itertools.chain(*ob))
        state_flatten = [] # 내부 튜플 제거...
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

        # choose action using state by policy
        action = self.policy.predict(state)     # predict의 return 은 Action의 형태를 가진다.
        return action
