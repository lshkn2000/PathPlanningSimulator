from collections import deque
from collections import namedtuple
from path_planning_simulator.sim.agent import Agent


class DynamicObstacle(Agent):
    def __init__(self, dy_obstacle_name="DynamicObstacle"):
        super(DynamicObstacle, self).__init__()
        self.name = dy_obstacle_name

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")

        # set state information
        state = [self.self_state_w_goal, ob]  # [(장애물 자신의 정보), [(다른 장애물들 정보 ; x, y, vx, vy, radius)]]

        # choose action using state by policy
        action = self.policy.predict(state)

        return action


class StaticObstacle(Agent):
    def __init__(self):
        super(StaticObstacle, self).__init__()
        '''
        if rectangle:
        +------------------+
        |                  |
      height    (x,y)      |
        |                  |
        +------ width -----+
        '''
        self.rectangle = True
        self.width = None
        self.height = None

    def act(self, ob):
        action = deque([0,0], maxlen=2)
        return action

    def set_rectangle(self, width, height):
        self.width = width
        self.height = height

        self.vx = 0
        self.vy = 0
        self.gx = None
        self.gy = None
        self.radius = 0
        self.v_pref = 0

    @property
    def self_state_wo_goal_rectangle(self):
        return self.px, self.py, self.width, self.height
