import itertools
import numpy as np


class BasicState():
    def __init__(self):
        pass

    def basic_state_function(self, ob):
        # Robot Info
        robot_info = ob[:7]
        robot_position = robot_info[0:2]
        robot_velocity = robot_info[2:4]
        robot_goal = robot_info[4:6]
        robot_size = robot_info[6]

        # Relative Goal
        robot_info[4:6] = robot_goal - robot_position

        # Dynamic Obstacle Info
        obstacles_info = ob[7:]
        obstacles_num = len(obstacles_info) // 5  # dynamic obstacle : [(px, py, vx, vy, radius)]
        obstacles = obstacles_info.reshape((-1, 5))
        relative_state_dy_obstacle = [(obstacle[0] - robot_position[0], obstacle[1] - robot_position[1],
                                       obstacle[2] - robot_velocity[0], obstacle[3] - robot_velocity[1],
                                       obstacle[4])
                                      for obstacle in obstacles]

        state = [robot_info] + [relative_state_dy_obstacle]

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

        return np.array(state_flatten)



