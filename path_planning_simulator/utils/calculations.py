import itertools
from typing import List
import numpy as np
from scipy.special import erf


def rotate2D(matrix, theta):
    # theta -> radian
    rot_theta = theta / (2 * np.pi)
    rot_theta = (rot_theta - np.trunc(rot_theta)) * (2 * np.pi)
    delta_theta = rot_theta - np.pi / 2

    delta_theta = -delta_theta      # 양수이면 반시계반향 음수이면 시계방향

    cos, sin = np.cos(delta_theta), np.sin(delta_theta)
    R = np.array([[cos, -sin], [sin, cos]])
    rotated = np.matmul(R, matrix)
    return float(rotated[0]), float(rotated[1])


def gaussian_distribution(x, mu, sigma):
    # x 는 속도 벡터의 방향에 대한 정보
    # 확률밀도 함수
    y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2) / (2 * sigma**2)
    # 누적 분포 함수
    y_cum = 0.5 * (1 + erf((x-mu)/(np.sqrt(2 * sigma**2))))
    return y


def state_flattening(ob: List):
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
    return state