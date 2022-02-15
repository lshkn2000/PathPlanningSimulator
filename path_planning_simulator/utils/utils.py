import numpy as np


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