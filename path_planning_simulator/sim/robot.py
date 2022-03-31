from collections import deque
import numpy as np
import torch

from path_planning_simulator.sim.agent import Agent
from path_planning_simulator.utils.state_function_engineering.basic_state_model import BasicState
from path_planning_simulator.utils.state_function_engineering.grid_based_state_model import GridBasedState
# from path_planning_simulator.utils.world_model.convolution import CNNModel # gridmap 을 위해서 만드는 중이다. 이건 CNNVAE에 적용하기 위함


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Robot(Agent):
    def __init__(self, cartesian=True, detection_scope=float("inf"), robot_name="Robot", state_engineering="Basic"):
        """
        :param cartesian:
                Robot moves according to cartesian coordinate,
            else :
                Polar coodrinate
        :param detection_scope:
            Range that robot detect obstacles
        :param robot_name:
            Info about robot name to distinguish other agents
        :param state_engineering:
            [Basic, GridMap, NonGridMap]
            if Basic :
                robot get relative positions and velocities about obstacles,
            elif GridMap:
                robot get information images about obstacles
            elif NonGridMap:
                robot get graph information about obstacles
        """
        super(Robot, self).__init__()
        self.name = robot_name

        self.cartesian = cartesian

        if self.cartesian:
            self.theta = 0          # Ignore direction
        else:
            self.theta = np.pi/2    # Start direction

        self.action = deque([None, None], maxlen=2)

        self.detection_scope = detection_scope

        self.state_engineering = state_engineering

        if self.state_engineering == "Basic":
            self.state_function = BasicState()
        elif self.state_engineering == "GridMap":
            self.state_function = GridBasedState()
        # elif self.state_engineering == "NonGridMap":
        #     self.state_function = NonGridBasedState()
        self.test_state = BasicState()

        # self.cnn_model = CNNModel()

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
        ob_length = ob.size
        print(f"before state : {state}")

        ############## State Engineering ##############
        # Observation Customization
        # 1. Basic model
        if self.state_engineering == "Basic":
            state = self.state_function.basic_state_function(ob)
            # State Encoder

        # 2. Grid based model
        # Relative Coordinate State information with image
        elif self.state_engineering == "GridMap":
            state = self.state_function.grid_based_state_function(ob, robot_detection_scope_radius=2.5,
                                                                  detection_scope_resolution=0.1, map_size=(10, 10))
            # State Encoder
            # state_img = np.array([state])
            # torched_state = torch.flip(torch.from_numpy(state_img), dims=(0,))
            # torched_state = torch.tensor(torched_state, dtype=torch.float32)
            # torched_state = torched_state.view(-1, 1, state.shape[0], state.shape[1])
            # self.cnn_model.model_setting((state.shape[0], state.shape[1]), [256, 128], ob_length, 1, 10)
            # state = self.cnn_model(torched_state).to(device)


        # 3. Non grid based model
        # elif self.state_engineering == "NonGridMap":
        #     self.state_function = NonGridBasedState()
        #     # State Encoder

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
