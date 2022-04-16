import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import torch
from PIL import Image
import time

from path_planning_simulator.sim.agent import Agent
from path_planning_simulator.utils.state_function_engineering.basic_state_model import BasicState
from path_planning_simulator.utils.state_function_engineering.grid_based_state_model import GridBasedState


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class Robot(Agent):
    def __init__(self, cartesian=True, robot_name="Robot", state_engineering="Basic"):
        """
        :param cartesian:
                Robot moves according to cartesian coordinate,
            else :
                Polar coodrinate
        :param robot_name:
            Info about robot name to distinguish other agents
        :param state_engineering:
            [Basic, GridMap, NonGridMap]
            if Basic :
                robot get relative positions and velocities about obstacles,
            elif VAE:
                robot get information images about obstacles using VAE latent variables

        """
        super(Robot, self).__init__()
        self.name = robot_name

        self.cartesian = cartesian

        if self.cartesian:
            self.theta = 0          # Ignore direction
        else:
            self.theta = np.pi/2    # Start direction

        self.action = deque([None, None], maxlen=2)

        self.state_engineering = state_engineering

        if self.state_engineering == "Basic":
            self.state_function = BasicState()
        elif self.state_engineering == "VAE":
            self.state_function = GridBasedState(is_relative=False)

        self.img_transformer = None
        self.detection_scope = None
        self.detection_scope_resolution = None
        self.img_plot_size = None
        self.vae_model = None

    def set_vae_model(self, vae_model):
        self.vae_model = vae_model

    def set_state_img_param(self, img_transformer, detection_scope, resolution, plot_size):
        self.img_transformer = img_transformer
        self.detection_scope = detection_scope
        self.detection_scope_resolution = resolution
        self.img_plot_size = plot_size

    def act(self, ob):
        if self.policy is None:
            raise AttributeError("Need to set policy!")
        """
        Robot Observation constructed with 3 categories 
        0. Robot info : [px, py, vx, vy, gx, gy, radius]
        1. Dynamic Obstacle info : [(px, py, vx, vy, radius), ...]
        2. Static Obstacle info [(px, py, width, height)]
        """
        state = self.make_encoding_state(ob)

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

    def make_encoding_state(self, ob):
        ############## State Engineering ##############
        # Observation Customization
        # 1. Basic model
        if self.state_engineering == "Basic":
            state = self.state_function.basic_state_function(ob)
            return state

        # 2. Grid based model
        # Relative Coordinate State information with image
        elif self.state_engineering == "VAE":
            start_drawing = time.time()
            grid_state, (f, ax) = self.state_function.grid_based_state_function(ob,
                                                                                self.detection_scope,
                                                                                self.detection_scope_resolution,
                                                                                self.img_plot_size)

            print(f"drawing_time : {time.time() - start_drawing}")

            start_img_conversion =time.time()
            f.canvas.draw()
            img = np.array(f.canvas.renderer._renderer) # RGBA : 4 channel
            img = Image.fromarray(img.astype('uint8'))
            img = img.convert("RGB")    # RGB : 3 channel
            plt.close(f)
            transformed_img = self.img_transformer(img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            print(f"img_conversion_time : {time.time() - start_img_conversion}")

            start_z = time.time()
            _, _, _, z = self.vae_model(transformed_img)
            state = z.cpu().detach().numpy()
            print(f"z_time : {time.time() - start_z}")
            return state

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
