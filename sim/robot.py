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

        # set state information
        state = ob
        # choose action using state by policy
        action = self.policy.predict(state)     # predict의 return 은 Action의 형태를 가진다.
        return action
