from collections import deque
import numpy as np

from path_planning_simulator.sim.agent import Agent


class Robot(Agent):
    def __init__(self, cartesian=True, detection_scope=float("inf"), robot_name="Robot"):
        super(Robot, self).__init__()
        self.name = robot_name

        self.cartesian = cartesian

        if self.cartesian:
            self.theta = 0          # Ignore direction
        else:
            self.theta = np.pi/2    # Start direction

        self.action = deque([None, None], maxlen=2)

        self.detection_scope = detection_scope

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")

        # 로봇이 받는 상태정보는 3가지의 리스트로 구성되어있다
        # 0. Robot info : [px, py, vx, vy, gx, gy, radius]
        # 1. Dynamic Obstacle info : [(px, py, vx, vy, radius), ...]
        # 2. Static Obstacle info [(px, py, width, height)]

        # set state information
        # Observation Customization
        state = ob
        # state = self.policy.lstm.custom_state_for_lstm(state)
        # state = self.policy.featured_state(state)

        # Action for State
        action = self.policy.predict(state)

        if isinstance(action, np.ndarray):
            self.action[0] = action[0]
            self.action[1] = action[1]
            return self.action
        else:
            print("action : ", action)
            print("action type : {}".format(type(action)))
            raise Exception("Check the action dtype. Dtype must be Numpy")

    def step(self, action):
        # Cartesian Coordinate Policy [Vx, Vy]
        if self.cartesian:
            # noise
            action += np.random.normal(0.0, 0.1, size=2)  # normal(mean, std, action_space)
            action = action.clip(-1, 1)  # action scale (min, max)

            self.vx = action[0]
            self.vy = action[1]
            self.px = self.px + action[0] * self.time_step
            self.py = self.py + action[1] * self.time_step
        # Polar Coordinate Policy [W, V] -> [Vx, Vy]
        else:
            # Get Orientation
            self.theta = self.theta + (action[0] * np.pi) * self.time_step  # input -pi ~ pi (rad/s)
            # scope angle to -2pi ~ 2pi
            rot_delta_theta = self.theta / (2 * np.pi)
            rot_delta_theta = (rot_delta_theta - np.trunc(rot_delta_theta)) * (2 * np.pi)
            # scope 0 ~ 2pi
            rot_delta_theta = (2 * np.pi + rot_delta_theta) * (rot_delta_theta < 0) + rot_delta_theta * (
                    rot_delta_theta > 0)

            # noise
            action += np.random.normal(0.0, 0.1, size=2)  # normal(mean, std, action_space)
            action = action.clip(-1, 1)  # action scale (min, max)

            # Convert [W, V] to [Vx, Vy]
            self.vx = action[1] * np.cos(rot_delta_theta)
            self.vy = action[1] * np.sin(rot_delta_theta)
            self.px = self.px + action[0] * self.time_step
            self.py = self.py + action[1] * self.time_step

    def store_trjectory(self, state, action, reward, new_state, is_terminal):
        self.policy.store_trajectory(state, action, reward, new_state, is_terminal)
