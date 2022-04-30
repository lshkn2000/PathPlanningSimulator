from collections import deque
import numpy as np
import torch

from PathPlanningSimulator_new.path_planning_simulator.sim.agent import Agent


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Robot(Agent):
    def __init__(self, robot_name="Robot", is_relative=False):
        super(Robot, self).__init__()
        self.name = robot_name
        self.is_relative = is_relative

        self.action = deque([None, None], maxlen=2)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")
        """
        Robot Observation constructed with 3 categories 
        0. Robot info : [px, py, vx, vy, gx, gy, radius]
        1. Dynamic Obstacle info : [(px, py, vx, vy, radius), ...]
        2. Static Obstacle info [(px, py, width, height)]
        """
        state = ob

        # Action for State
        action = self.policy.predict(state)

        if isinstance(action, np.ndarray):
            self.action[0] = action[0]
            self.action[1] = action[1]
            return np.array(self.action)
        else:
            print("action : ", action)
            print("action type : {}".format(type(action)))
            raise Exception("Check the action dtype. Dtype must be Numpy")

    def step(self, action):
        # Cartesian Coordinate Policy [Vx, Vy]
        # noise
        action += np.random.normal(0.0, 0.1, size=2)  # normal(mean, std, action_space)
        action = action.clip(-1, 1)  # action scale (min, max)

        self.vx = action[0]
        self.vy = action[1]
        self.px = self.px + action[0] * self.time_step
        self.py = self.py + action[1] * self.time_step

    def store_trjectory(self, state, action, reward, new_state, is_terminal):
        self.policy.store_trajectory(state, action, reward, new_state, is_terminal)
