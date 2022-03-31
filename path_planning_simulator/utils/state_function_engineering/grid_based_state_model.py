import numpy as np
from scipy.stats import norm, multivariate_normal
from matplotlib import pyplot as plt


class GridBasedState():
    def __init__(self):
        self.grid_heatmap_logs = []

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

        grid_map_size = 2 * robot_detection_scope_radius    # grid map 한변 길이 = 2 * 반지름

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
        total_obstacles_position = [obstacle[:2] - robot_position for obstacle in obstacles]
        total_obstacles_position = np.array(total_obstacles_position)

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
            scoped_obstacles_position = [obstacle[:2] - robot_position for obstacle in scoped_obstacles]  # [px, py, value]

            scoped_obstacles_position = np.array(scoped_obstacles_position)

            # value(information) at obstacle position
            position_values = np.array([[1] for _ in range(len(scoped_obstacles))])
            obstacle_pose_with_info = np.concatenate((scoped_obstacles_position, position_values), axis=1)
        else:
            # 탐지 범위내에서 장애물을 감지하지 못하면 grid의 모든 위치의 value가 0인 값이다.
            scoped_obstacles_position = np.array([])
            obstacle_pose_with_info = None

        gmap, minx, maxx, miny, maxy = self.gaussian_grid_map(scoped_obstacles_position, grid_map_size, xyresolution=detection_scope_resolution, std=[0.5, 0.5])
        # self.grid_heatmap_logs.append(gmap)

        rot_gmap = np.rot90(gmap, 1)  # cw rotation # left_top 이 (0,0) 이므로 image frame 에 맞추는 작업

        ###### PLOT ######
        # plt.cla()
        # plt.ioff() # 계속 그리는 작업을 중단하고 plt.show() 일때 전체 업데이트 해서 보여줌 <--> plt.ion() : 디폴트
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #                              lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.rcParams["figure.figsize"] = map_size
        #
        # # 시각화
        # self.draw_heatmap(robot_position, gmap, minx, maxx, miny, maxy, xyresolution=detection_scope_resolution)
        #
        # # 모든 장애물의 world coord 위치 표시
        # for obstacle_position in total_obstacles_position:
        #     plt.plot(obstacle_position[0], obstacle_position[1], 'ob')
        # # 좌표계 선택에 따른 표현 방법
        # plt.plot(robot_goal[0] - robot_position[0], robot_goal[1] - robot_position[1], 'or')
        # plt.plot(0, 0, "og") # 로봇이 기준
        # # plt.pause(0.1) # 해당시간만큼 이미지를 보여주고 꺼짐
        # plt.show()  # 계속 이미지를 보여줌

        return rot_gmap

    def gaussian_grid_map(self, obstacles_positions, grid_map_size, xyresolution, std):
        # map size 가 NxN의 정사각형을 가정
        minx = -round(grid_map_size / 2)
        maxx = round(grid_map_size / 2)
        miny = -round(grid_map_size / 2)
        maxy = round(grid_map_size / 2)

        xw = int(round((maxx - minx) / xyresolution))
        yw = int(round((maxy - miny) / xyresolution))

        xyreso = xyresolution

        # gmap = [[0.0 for i in range(yw)] for i in range(xw)]
        x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso), slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]
        xy = np.column_stack([x.ravel(), y.ravel()])

        gaussian_list = []
        for obstacle_position in obstacles_positions:
            k = multivariate_normal(mean=obstacle_position, cov=np.array(std))
            gaussian_list.append(k)

        if gaussian_list:
            z = np.array(sum(item.pdf(xy) for item in gaussian_list))
        else:
            z = np.array([])

        if z.size != 0:
            gmap = z.reshape((x.shape[0], y.shape[0]))
        else:
            gmap = np.zeros((x.shape[0], y.shape[0]))

        return gmap, minx, maxx, miny, maxy

    def draw_heatmap(self, robot_position, data, minx, maxx, miny, maxy, xyresolution):
        xyreso = xyresolution

        x, y = np.mgrid[slice(minx - xyreso / 2.0, maxx + xyreso / 2.0, xyreso), slice(miny - xyreso / 2.0, maxy + xyreso / 2.0, xyreso)]

        plt.pcolor(x, y, data, vmax=1.0, cmap=plt.cm.Blues)
        plt.axis("equal")
