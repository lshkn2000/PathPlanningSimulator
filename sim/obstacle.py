from collections import namedtuple
from sim.agent import Agent


Action = namedtuple('Action', ['vx', 'vy'])


class DynamicObstacle(Agent):
    def __init__(self, dy_obstacle_name="DynamicObstacle"):
        super(DynamicObstacle, self).__init__()
        self.name = dy_obstacle_name

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")

        # set state information
        dir_x = self.gx - self.px
        dir_y = self.gy - self.py

        state = [[dir_x, dir_y], ob]    # 동적 장애물에 정책에는 각 장애물의 목적지로의 방향 벡터 정보를 추가함.

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
        action = Action(vx=0, vy=0)
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
