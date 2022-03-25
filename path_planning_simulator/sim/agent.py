import abc
import numpy as np

class Agent(object):
    def __init__(self):
        self.name = None
        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.gx = None
        self.gy = None
        self.radius = None
        self.v_pref = None

        self.cartesian = True
        self.theta = 0

        self.policy = None  # 학습 알고리즘 클래스를 가져오기

        self.time_step = None

        self.goal_offset = 0

    def set_goal_offset(self, offset):
        self.goal_offset = offset

    def set_policy(self, policy):
        self.policy = policy

    def set_agent_attribute(self, px, py, vx, vy, gx, gy, radius, v_pref=1, time_step=0.01):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.radius = radius
        self.v_pref = v_pref

        self.time_step = time_step

    @property
    def position(self):
        return self.px, self.py     # return tuple

    @property
    def velocity(self):
        return self.vx, self.vy     # return tuple # if non-holonomic, vx 는 각속도, vy 는 선속도

    @property
    def goal(self):
        return self.gx, self.gy     # return tuple

    @property
    def size(self):
        return self.radius

    @property
    def velocity_preference(self):
        return self.v_pref

    @property
    def self_state_w_goal(self):
        return self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.radius
        # return self.px, self.py, self.vx, self.vy, self.gx - self.px, self.gy - self.py, self.radius

    @property
    def self_state_wo_goal(self):
        return self.px, self.py, self.vx, self.vy, self.radius

    @property
    def info(self):
        print("============================================")
        print("Agent Name : ", self.name)
        print("policy : ", self.policy.__class__.__name__)
        print("============================================")
        return

    @abc.abstractmethod
    def act(self, ob):
        '''
        first : get observation information and set state information
        second : select action by policy
        '''
        return

    def step(self, action):
        self.vx = action[0]
        self.vy = action[1]
        self.px = self.px + action[0] * self.time_step
        self.py = self.py + action[1] * self.time_step

    def reach_goal(self):
        px, py = self.position
        gx, gy = self.goal

        l2 = pow(px - gx, 2) + pow(py - gy, 2)
        l2_norm = pow(l2, 0.5)

        return l2_norm < self.radius + self.goal_offset

    def update_policy(self):
        # 학습 네트워크 업데이트
        pass

    def store_transition(self):
        # 자신의 행동에 대한 기록 저장
        pass


