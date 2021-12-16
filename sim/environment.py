import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.lines as mlines
import matplotlib.animation as animation


class Environment(object):
    def __init__(self, time_step, time_limit):
        '''
        환경 맵 세팅(main)
        장애물 배치(main) -> 세팅(Env) : 완료
            동적 장애물은 원
            정적 장애물은 사각형
        로봇 배치(main) -> 세팅(Env) : 완료

        run sim -> loop 돌면서 스텝 진행
        '''
        # map 크기
        self.square_width = 10
        self.square_height = 10

        # 로봇과 장애물 소환
        self.robot = None
        self.dy_obstacles = []
        self.st_obstacles = []

        # reset() 할 때 초기 위치 설정
        self.init_position = None
        self.init_velocity = None
        self.init_goal = None

        # 한 에피소드당 행동 정보 기록용. reset()하면 내용 삭제됨
        self.dy_obstacles_positions = []
        self.robot_position = None

        # 한 에피소드당 스텝 시간 측정용
        self.time_step = time_step
        self.global_time = None
        self.time_limit = time_limit

        # 충돌 거리 설정
        self.safe_distance = 0.3

    def set_robot(self, robot):
        self.robot = robot
        self.init_position = robot.position
        self.init_velocity = robot.velocity
        self.init_goal = robot.goal
        self.robot_position = []

    def set_dynamic_obstacle(self, obstacle):
        # 장애물 수만큼 불러주어야 함
        self.dy_obstacles.append(obstacle)
        # 각 장애물 마다 위치 기록을 저장하므로 2차원 리스트 적용
        self.dy_obstacles_positions.append([])

    def set_static_obstacle(self, obstacle):
        # 장애물 수만큼 불러주어야 함
        self.st_obstacles.append(obstacle)

    @property
    def dy_obstacles_list(self):
        return self.dy_obstacles

    @property
    def st_obstacles_list(self):
        return self.st_obstacles

    def step(self, robot_action):
        # action and update position
        self.robot.step(robot_action)
        # 로봇 위치 정보 저장
        self.robot_position.append(self.robot.position)

        # vx, vy 정보를 이용하여 각 step 적용 용도로 사용
        dy_obstacles_actions = []

        for dy_obstacle in self.dy_obstacles:
            ob = [other_dy_obstacle.self_state_wo_goal for other_dy_obstacle in self.dy_obstacles if
                  other_dy_obstacle != dy_obstacle]  # tuple (px, py, vx, vy, radius)
            obstacle_action = dy_obstacle.act(ob)  # vx, vy
            dy_obstacles_actions.append(obstacle_action)

        for i, obstacle_action in enumerate(dy_obstacles_actions):
            # 각 장애물 행동 하고
            self.dy_obstacles[i].step(obstacle_action)
            # 장애물의 위치 정보 저장
            self.dy_obstacles_positions[i].append(self.dy_obstacles[i].position)

        self.global_time += self.time_step

        # collision check btw robot and dynamic obstacle
        '''
        장애물 사이의 충돌은 고려하지 않음
        '''
        collision = False
        reach_goal = False

        # 동적 장애물 충돌 검사
        closest_dist_of_dy_obs = float("inf")
        for i, dy_obstacle in enumerate(self.dy_obstacles):
            dx = dy_obstacle.px - self.robot.px
            dy = dy_obstacle.py - self.robot.py

            l2 = pow(dx, 2) + pow(dy, 2)
            l2_norm = pow(l2, 0.5)

            if closest_dist_of_dy_obs > l2_norm:
                closest_dist_of_dy_obs = l2_norm

            if closest_dist_of_dy_obs - self.robot.radius - dy_obstacle.radius < 0:
                collision = True
                break

        # 정적 장애물 충돌 검사
        closest_dist_of_st_obs = float("inf")
        for i, st_obstacle in enumerate(self.st_obstacles):
            # 사각형 장애물에 대한 충돌 검사
            if st_obstacle.rectangle:
                # 사각형의 좌상단, 우하단의 점을 기준으로 현재 로봇의 위치에 대한 최단 거리의 점을 클램핑으로 계산
                rect_left_floor = (st_obstacle.px - (st_obstacle.width / 2), st_obstacle.py + (st_obstacle.height / 2))
                rect_right_bottom = (st_obstacle.px + (st_obstacle.width / 2), st_obstacle.py - (st_obstacle.height / 2))

                clamped_x = max(rect_left_floor[0], min(rect_right_bottom[0], self.robot.px))
                clamped_y = max(rect_right_bottom[1], min(rect_left_floor[1], self.robot.py))

                # 투영 점 (= 로봇과 사각형 사이의 최단 거리의 점) 과 로봇의 위치에 대한 최단 거리 계산
                dx = self.robot.px - clamped_x
                dy = self.robot.py - clamped_y

                l2 = pow(dx, 2) + pow(dy, 2)
                l2_norm = pow(l2, 0.5)

                # 최단 거리가 원의 반지름보다 크면 충돌하지 않고 원의 반지름보다 작으면 충돌
                if l2_norm < self.robot.radius:
                    collision = True
                    print("collision!")
                    break

            else:
                # 원 장애물에 대한 충돌 검사
                dx = st_obstacle.px - self.robot.px
                dy = st_obstacle.py - self.robot.py

                l2 = pow(dx, 2) + pow(dy, 2)
                l2_norm = pow(l2, 0.5)

                if closest_dist_of_st_obs > l2_norm:
                    closest_dist_of_st_obs = l2_norm

                if closest_dist_of_st_obs - self.robot.radius - st_obstacle.radius < 0:
                    collision = True
                    break

        # check reaching the goal
        reach_goal = self.robot.reach_goal()

        # reward setting
        '''
        조건 : time, collision, reach_goal 
        '''
        if reach_goal:
            reward = 1
            done = True
            info = None
        elif collision:
            reward = -1
            done = True
            info = None
        elif self.global_time >= self.time_limit - 1:
            reward = -1
            done = True
            info = None
        else:
            reward = 0
            done = False
            info = None

        # next_step_ob
        next_robot_ob = [robot_state_data for robot_state_data in self.robot.self_state_w_goal]
        next_dy_obstacle_ob = [dy_obstacle.self_state_wo_goal for dy_obstacle in self.dy_obstacles]
        next_st_obstacle_ob = [st_obstacle.self_state_wo_goal_rectangle for st_obstacle in self.st_obstacles]
        next_ob = [next_robot_ob] + [next_dy_obstacle_ob] + [next_st_obstacle_ob]

        return next_ob, reward, done, info

    def reset(self, random_position=False, random_goal=False):
        # 에피소드 실행 시간 초기화
        self.global_time = 0

        # 로봇, 장애물의 위치 정보 초기화
        self.robot_position = []
        self.dy_obstacles_positions = [[] for _ in range(len(self.dy_obstacles_list))]

        # 로봇 위치, 속도, 목적지 초기화
        # random 적용
        sign = 1 if np.random.random() > 0.5 else -1
        if random_position:
            robot_px = np.random.random() * (self.square_width * 0.5) * sign
            robot_py = (np.random.random() - 0.5) * self.square_height
            self.robot.px = robot_px
            self.robot.py = robot_py
            self.robot.vx = 0
            self.robot.vy = 0
        else:
            self.robot.px, self.robot.py = self.init_position
            self.robot.vx, self.robot.vy = self.init_velocity

        if random_goal:
            robot_gx = np.random.random() * (self.square_width * 0.5) * sign
            robot_gy = (np.random.random() - 0.5) * self.square_height
            self.robot.gx = robot_gx
            self.robot.gy = robot_gy
        else:
            self.robot.gx, self.robot.gy = self.init_goal

        # 장애물 위치 초기화
        self.generate_random_position()

        # 로봇과 장애물의 상태 정보 출력
        robot_ob = [robot_state_data for robot_state_data in self.robot.self_state_w_goal]
        dy_obstacle_ob = [dy_obstacle.self_state_wo_goal for dy_obstacle in self.dy_obstacles]
        st_obstacle_ob = [st_obstacle.self_state_wo_goal_rectangle for st_obstacle in self.st_obstacles]
        ob = [robot_ob] + [dy_obstacle_ob] + [st_obstacle_ob]

        return ob

    def generate_random_position(self):
        '''
        로봇이 우선 배치되고 해당 배치된 위치를 기준으로 장애물 배치
        '''

        robot_px, robot_py = self.robot.position
        robot_gx, robot_gy = self.robot.goal
        no_collision_distance = self.safe_distance

        for i, dy_obstacle in enumerate(self.dy_obstacles):
            # obstacles 의 속성들(v_pref, radius) 변경
            dy_obstacle.v_pref = np.random.uniform(0.5, 1.5)
            dy_obstacle.radius = np.random.uniform(0.3, 0.5)

            # 장애물 위치 변경 및 목적지 설정
            # 로봇의 위치에서 일정 거리 떨어진 곳에서 랜덤 생성
            sign = 1 if np.random.random() > 0.5 else -1  # 장애물의 위치와 목적지 위치를 서로 반대 방향에 설정하기 위함
            while True:
                # 맵의 중심을 (0,0)으로 기준 잡았을 경우
                obstacle_px = np.random.random() * (self.square_width * 0.5) * sign
                obstacle_py = (np.random.random() - 0.5) * self.square_height

                # 로봇과 랜덤 배치된 장애물의 거리를 재고 안전거리 이상 떨어져 있으면 장애물 위치 설정
                if np.linalg.norm((robot_px - obstacle_px,
                                   robot_py - obstacle_py)) - dy_obstacle.radius - self.robot.radius > no_collision_distance:
                    dy_obstacle.px = obstacle_px
                    dy_obstacle.py = obstacle_py
                    break

            while True:
                # 맵의 중심을 (0,0)으로 기준 잡았을 경우
                obstacle_gx = np.random.random() * (self.square_width * 0.5) * -sign
                obstacle_gy = (np.random.random() - 0.5) * self.square_height

                # 장애물의 목적지가 로봇의 목적지와 안전거리 이상 떨어져 있게 목적지 위치 설정
                if np.linalg.norm((robot_gx - obstacle_gx,
                                   robot_gy - obstacle_gy)) - dy_obstacle.radius - self.robot.radius > no_collision_distance:
                    dy_obstacle.gx = obstacle_gx
                    dy_obstacle.gy = obstacle_gy
                    break

    def render(self):
        # color setting
        robot_color = 'black'
        static_obstacle_color = 'yellow'
        dynamic_obstacle_color = 'blue'
        goal_color = 'red'

        fig, ax = plt.subplots(figsize=(self.square_width, self.square_height))
        ax.set_xlim(-int(self.square_width / 2), int(self.square_width / 2))
        ax.set_ylim(-int(self.square_width / 2), int(self.square_width / 2))
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # 에피소드, 스텝 시간 표시
        step_cnt = plt.text(self.square_width / 2, self.square_height / 2, 'Step : {}'.format(0), fontsize=16)
        ax.add_artist(step_cnt)

        # 초기 로봇 그리기
        robot_circle = Circle(self.robot_position[0], self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot_circle)

        # 목적지 그리기
        goal_x, goal_y = self.robot.goal
        goal = mlines.Line2D([goal_x], [goal_y], color=goal_color, marker='*', linestyle='None', markersize=15,
                             label='Goal')
        ax.add_artist(goal)

        # j개의 초기 동적 장애물 그리기
        dy_obstacle_circle_list = []
        for j, dy_obstacle in enumerate(self.dy_obstacles_list):
            j_th_dy_obstacle_positions = self.dy_obstacles_positions[j]
            dy_obstacle_circle = Circle(j_th_dy_obstacle_positions[0], dy_obstacle.radius, fill=True,
                                        color=dynamic_obstacle_color)
            dy_obstacle_circle_list.append(dy_obstacle_circle)
            ax.add_artist(dy_obstacle_circle)

        # 정적 장애물 그리기
        for st_obstacle in self.st_obstacles_list:
            if st_obstacle.rectangle:
                # 사각형 정적 장애물
                x_for_rect = st_obstacle.px - (st_obstacle.width / 2)
                y_for_rect = st_obstacle.py - (st_obstacle.height / 2)
                st_obstacle_rectangle = Rectangle((x_for_rect, y_for_rect), st_obstacle.width, st_obstacle.height, angle=0.0)
                ax.add_artist(st_obstacle_rectangle)
            else:
                # 원형 정적 장애물
                st_obstacle_circle = Circle(st_obstacle.position, st_obstacle.radius, fill=True,
                                            color=static_obstacle_color)
                ax.add_artist(st_obstacle_circle)

        dy_obstacles_positions = self.dy_obstacles_positions

        def animate(frame):
            if frame == len(self.robot_position) - 1:

                print('steps done. closing!')
                # plt.ion()
                # plt.close(fig)

            else:
                # 로봇의 위치 기록을 기반으로 움직이기
                robot_circle.center = self.robot_position[frame]
                # 동적 장애물 위치 기록을 기반으로 움직이기
                for k, dy_obst in enumerate(dy_obstacle_circle_list):
                    k_th_dy_obst_positions = dy_obstacles_positions[k]
                    dy_obst.center = k_th_dy_obst_positions[frame]

            step_cnt.set_text('Step : {}'.format(frame+1))

        ani = animation.FuncAnimation(fig, animate, frames=len(self.robot_position))
        # print(self.dy_obstacles_positions[0])

        plt.show()

