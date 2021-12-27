from collections import namedtuple
from path_planning_simulator.policy.policy import Policy


class Linear(Policy):
    '''
    동적 장애물의 행동 설정을 위한 정책
    목적지 방향으로의 직진 이동
    '''
    def __init__(self, velocity=1):
        super(Linear, self).__init__()
        self.Action = namedtuple('Action', ['vx', 'vy'])
        self.velocity = velocity
        self.threshold = 0.1

    def predict(self, state):
        # State : 2개의 리스트로 구성
        # 0. 자기 자신의 정보 (self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.radius)
        # 1. 자기 자신 제외한 dynamic obstacles 의 (px, py, vx, vy, radius)

        # 목적지로의 방향벡터의 정규화
        self_state = state[0]
        dir_x = self_state[4] - self_state[0]
        dir_y = self_state[5] - self_state[1]

        l2 = pow(dir_x, 2) + pow(dir_y, 2)
        l2_norm = pow(l2, 0.5)

        norm_dir_x = dir_x / l2_norm
        norm_dir_y = dir_y / l2_norm

        if l2_norm < self.threshold:
            velocity = 0
        else:
            velocity = self.velocity

        # 목적지 방향으로 속도를 설정
        self.Action.vx = norm_dir_x * velocity
        self.Action.vy = norm_dir_y * velocity

        return self.Action

