from collections import namedtuple
from policy.policy import Policy


class Linear(Policy):
    '''
    동적 장애물의 행동 설정을 위한 정책
    목적지 방향으로의 직진 이동
    '''
    def __init__(self, default_velocity=1):
        super(Linear, self).__init__()
        self.Action = namedtuple('Action', ['vx', 'vy'])
        self.velocity = default_velocity
        self.threshold = 0.1

    def predict(self, state):
        # State : 2개의 리스트로 구성
        # 0. 현재 진행 방향 벡터 dir_x, dir_y
        # 1. 자기 자신 제외한 dynamic obstacles 의 (px, py, vx, vy, radius)

        print(state)

        # 목적지로의 방향벡터의 정규화
        direction_vector = state[0]
        dir_x = direction_vector[0]
        dir_y = direction_vector[1]

        l2 = pow(dir_x, 2) + pow(dir_y, 2)
        l2_norm = pow(l2, 0.5)

        norm_dir_x = dir_x / l2_norm
        norm_dir_y = dir_y / l2_norm

        if l2_norm < self.threshold:
            self.velocity = 0

        # 목적지 방향으로 속도를 설정
        self.Action.vx = norm_dir_x * self.velocity
        self.Action.vy = norm_dir_y * self.velocity

        return self.Action

