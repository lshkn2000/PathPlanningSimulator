import numpy as np
from collections import namedtuple
from path_planning_simulator.policy.policy import Policy


class Random(Policy):
    def __init__(self):
        super(Random, self).__init__()
        self.Action = namedtuple('Action', ['vx', 'vy'])

    def predict(self, state):
        self.Action.vx = np.random.uniform(-1, 1)
        self.Action.vy = np.random.uniform(-1, 1)

        return self.Action

