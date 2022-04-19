import time
import numpy as np
from scipy.stats import norm, multivariate_normal
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

class GridBasedState():
    def __init__(self, is_relative=False):
        self.grid_heatmap_logs = []
        self.is_relative = is_relative

    def grid_based_state_function(self, ob, robot_detection_scope_radius, detection_scope_resolution, map_size):
        """
        :param ob: robot :
            [px, py, vx, vy, gx, gy, radius] + obstacles : [px, py, vx, vy, radius]
        :param robot_detection_scope_radius:
            로봇이 장애물을 인식하는 탐지 반경 (m)
        :param detection_scope_resolution :
            grid map의 resolution, 작을수록 더 세밀함. 0.1 = 0.1 m 단위 분해능
        :param map_size: 맵의 크기 (m)
        :return:
            grid map image를 출력한다.
             img size width, height = (robot_detection_scope_radisu * 2) / detection_scope_resolution + 1
        """
        # Set scopping condition
        if robot_detection_scope_radius <= 0:
            is_scoping = False
        else:
            is_scoping = True

        grid_map_size = int(map_size[0])    # grid map 한변 길이 = 2 * 반지름

        # ob의 정보에서 로봇 정보와 obstacle 정보를 분리
        # Robot Info
        robot_info = ob[:7]
        robot_position = robot_info[0:2]
        robot_velocity = robot_info[2:4]
        robot_goal = robot_info[4:6]
        robot_size = robot_info[6]

        # Dynamic Obstacle Info
        obstacles_info = ob[7:]
        obstacles_num = len(obstacles_info) // 5    # dynamic obstacle : [(px, py, vx, vy, radius)]
        obstacles = obstacles_info.reshape((-1, 5))

        # 탐지 범위와 상관없이 장애물의 정보 추출
        if self.is_relative:
            total_obstacles_position_velocity = [np.append(np.append(obstacle[:2] - robot_position, obstacle[2:4] - robot_velocity), obstacle[4]) for obstacle in obstacles]
        else:
            total_obstacles_position_velocity = [np.append(np.append(obstacle[:2], obstacle[2:4]), obstacle[4]) for obstacle in obstacles]
        total_obstacles_position_velocity = np.array(total_obstacles_position_velocity)

        # robot의 탐지 범위 내의 장애물 정보만 추출
        scoped_obstacles = []
        for obstacle in obstacles:
            obstacle_position = obstacle[:2]    # px, py
            if is_scoping:   # detect obstacles in scope
                # 원형 탐지 범위
                # if np.linalg.norm(robot_position - obstacle_position) < robot_detection_scope_radius:
                #     scoped_obstacles.append(obstacle_position)
                # 사각형 탐지 범위
                if np.abs(robot_position[0] - obstacle_position[0]) < robot_detection_scope_radius and np.abs(robot_position[1] - obstacle_position[1]) < robot_detection_scope_radius:
                    scoped_obstacles.append(obstacle_position)
            else:            # detect all obstacles in map
                scoped_obstacles.append(obstacle_position)
        scoped_obstacles = np.array(scoped_obstacles)

        if scoped_obstacles.size != 0:
            # 탐지 범위 내에 있는 장애물에 대한 위치 grid map 생성
            # 로봇의 위치를 중심으로 장애물의 상대적인 위치를 표현하고 grid map 의 좌표계 중심으로 이동
            if self.is_relative:
                scoped_obstacles_position = [obstacle[:2] - robot_position for obstacle in scoped_obstacles]  # [px, py, value]
            else:
                scoped_obstacles_position = [obstacle[:2] for obstacle in scoped_obstacles]
            scoped_obstacles_position = np.array(scoped_obstacles_position)

            # value(information) at obstacle position
            position_values = np.array([[1] for _ in range(len(scoped_obstacles))])
            obstacle_pose_with_info = np.concatenate((scoped_obstacles_position, position_values), axis=1)
        else:
            # 탐지 범위내에서 장애물을 감지하지 못하면 grid의 모든 위치의 value가 0인 값이다.
            scoped_obstacles_position = np.array([])
            obstacle_pose_with_info = None

        # Drawing obstacle in OpenCV
        transformed_robot_position, tranformed_scale = self.get_cv2_coord_transform(np.array([robot_position]),
                                                                                    plot_size_real_val=(10, 10),
                                                                                    opencv_plot_size_pixel_val=(256, 256))
        transformed_robot_size = (robot_size * tranformed_scale).astype(np.uint8)

        # end point of velocity vector
        transformed_robot_velocity, _ = self.get_cv2_coord_transform(np.array([robot_velocity]),
                                                                  plot_size_real_val=(10, 10),
                                                                  opencv_plot_size_pixel_val=(256, 256))

        transformed_robot_goal, _ = self.get_cv2_coord_transform(np.array([robot_goal]),
                                                              plot_size_real_val=(10, 10),
                                                              opencv_plot_size_pixel_val=(256, 256))
        transformed_robot_goal_threshold = int(0.3 * tranformed_scale)

        tranformed_obstacles_position, _ = self.get_cv2_coord_transform(total_obstacles_position_velocity[:, 0:2],
                                                                        plot_size_real_val=(10, 10),
                                                                        opencv_plot_size_pixel_val=(256, 256))

        # end point of velocity vector
        tranformed_obstacles_velocitiy, _ = self.get_cv2_coord_transform(total_obstacles_position_velocity[:, 2:4],
                                                                         plot_size_real_val=(10, 10),
                                                                         opencv_plot_size_pixel_val=(256, 256))
        tranformed_obstacles_velocitiy -= int(256/2) # opencv_plot_size_pixel_val / 2. for calculate length
        tranformed_obstacles_radius = (total_obstacles_position_velocity[:, -1] * tranformed_scale).astype(np.uint8).reshape(-1, 1)

        map = np.zeros((256, 256, 3), np.uint8)

        for obst_posi_n_rad in zip(tranformed_obstacles_position, tranformed_obstacles_radius):
            map = cv2.circle(map, obst_posi_n_rad[0], obst_posi_n_rad[1].item(), GREEN, -1)

        # Draw velocity range and warning sign
        # failed to drawing...
        # for ob_posi_n_vel in zip(tranformed_obstacles_position, tranformed_obstacles_velocitiy):
        #     #
        #     ob_posi = ob_posi_n_vel[0]
        #     ob_vel = ob_posi_n_vel[1]
        #     obst_vel_length = np.sqrt(ob_vel[0] ** 2 + ob_vel[1] ** 2)  # velocity wedge length
        #     ob_vel_angle = np.arctan2(ob_vel[1], ob_vel[0])  # velocity angle
        #     if ob_vel_angle < 0:
        #         ob_vel_angle = 2 * np.pi + ob_vel_angle
        #
        #     # if robot in obstacle velocity length range, change color
        #     ob_vel_angle_range = np.pi / 3
        #     distance_btw_robot_obstacle = np.sqrt((int(transformed_robot_position[0][0]) - int(ob_posi[0])) ** 2 +
        #                                           (int(transformed_robot_position[0][1]) - int(ob_posi[1])) ** 2)
        #     relative_angle_btw_robot_obstacle = np.arctan2(int(transformed_robot_position[0][1])-int(ob_posi[1]),
        #                                                    int(transformed_robot_position[0][0])-int(ob_posi[0]))
        #     if relative_angle_btw_robot_obstacle < 0:
        #         relative_angle_btw_robot_obstacle = 2 * np.pi + relative_angle_btw_robot_obstacle
        #     if obst_vel_length > distance_btw_robot_obstacle - transformed_robot_size and \
        #             (ob_vel_angle - ob_vel_angle_range < relative_angle_btw_robot_obstacle < ob_vel_angle + ob_vel_angle_range):
        #         face_color = (0, 150, 150)
        #     else:
        #         face_color = (0, 200, 0)
        #
        #     start_angle = np.rad2deg(ob_vel_angle - ob_vel_angle_range)
        #     end_angle = np.rad2deg(ob_vel_angle + ob_vel_angle_range)
        #
        #     map = cv2.ellipse(map, (ob_posi[0], ob_posi[1]), (int(obst_vel_length), int(obst_vel_length)), 0, int(start_angle), int(end_angle), face_color, -1)


        # Draw robot and target
        map = cv2.circle(map, transformed_robot_position[0], transformed_robot_size.item(), RED, -1)
        map = cv2.circle(map, transformed_robot_goal[0], transformed_robot_goal_threshold, BLUE, -1)
        # failed to drawing
        # map = cv2.arrowedLine(map, transformed_robot_position[0], transformed_robot_position[0] + transformed_robot_velocity[0], RED, 3)

        # print("end time : {}".format(time.time() - start_opencv))
        # cv2.imshow('test', map)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        return map

    def get_cv2_coord_transform(self, obstacles_positions, plot_size_real_val=(10, 10), opencv_plot_size_pixel_val=(256, 256)):
        # map size 가 NxN의 정사각형을 가정
        minx = -round(plot_size_real_val[0] / 2)
        maxx = round(plot_size_real_val[0] / 2)
        miny = -round(plot_size_real_val[1] / 2)
        maxy = round(plot_size_real_val[1] / 2)

        scale = opencv_plot_size_pixel_val[0] / plot_size_real_val[0]

        pts1 = np.float32([[minx, maxy], [minx, miny], [maxx, miny], [maxx, maxy]])
        pts2 = np.float32([[0, 0],
                           [0, opencv_plot_size_pixel_val[1]],
                           [opencv_plot_size_pixel_val[0], opencv_plot_size_pixel_val[1]],
                           [opencv_plot_size_pixel_val[0], 0]])

        map_to_opencv_M = cv2.getPerspectiveTransform(pts1, pts2)

        # make matrix for transpose
        transpose_maxtrix_one_value_ = np.ones((len(obstacles_positions), 1))
        obstacles_positions = np.hstack((obstacles_positions, transpose_maxtrix_one_value_))
        obstacles_positions = obstacles_positions.transpose()

        # transpose
        transposed_obstacles_positions = np.matmul(map_to_opencv_M, obstacles_positions).transpose()
        #
        transposed_obstacles_positions = np.delete(transposed_obstacles_positions, -1, axis=1)   # delete np column
        transposed_obstacles_positions = transposed_obstacles_positions.astype(np.uint8)
        return transposed_obstacles_positions, scale
