# from IPython.display import clear_output
# %matplotlib notebook
from sim.environment import Environment
from sim.robot import Robot
from sim.obstacle import DynamicObstacle, StaticObstacle
from policy.random import Random
from policy.linear import Linear


def run_sim(env, episodes=1, render=True):
    dt = env.time_step

    # 각 로봇, 동적 장애물 행동 취하기
    # 에피소드 실행
    # env.reset(random_position=False, random_goal=False)
    # action =
    # ob, reward, done, info = step(action)

    print(env.robot.info)
    print(env.dy_obstacles[0].info)

    for i_episode in range(episodes):
        ob = env.reset(random_position=False, random_goal=False)
        for t in range(50):
            action = env.robot.act(ob)
            next_ob, reward, done, info = env.step(action)
            if done:
                break
        print("{} episode".format(i_episode + 1))

        env.render()


if __name__ == "__main__":
    # 환경 변수 설정
    time_step = 0.1
    time_limit = 10

    # 환경 소환
    env = Environment(time_step=time_step, time_limit=time_limit)

    # 로봇 소환
    robot = Robot()
    robot_init_position = {"px":0, "py":0, "vx":0, "vy":0, "gx":0, "gy":4}
    robot.set_agent_attribute(px=0, py=0, vx=0, vy=0, gx=0, gy=4, radius=0.3, v_pref=1, time_step=time_step)
    # 로봇 정책(행동 규칙) 세팅
    robot_policy = Random()
    robot.set_policy(robot_policy)

    # 장애물 소환
    # 동적 장애물
    dy_obstacle_num = 10
    dy_obstacles = [None] * dy_obstacle_num
    for i in range(dy_obstacle_num):
        dy_obstacle = DynamicObstacle()

        # 초기 소환 위치 설정
        # 이것도 함수로 한번에 설정할 수 있도록 변경할 것
        dy_obstacle.set_agent_attribute(px=2, py=2, vx=0, vy=0, gx=-2, gy=-2, radius=0.3, v_pref=1, time_step=time_step)
        # 동적 장애물 정책 세팅
        dy_obstacle_policy = Linear()
        dy_obstacle.set_policy(dy_obstacle_policy)

        dy_obstacles[i] = dy_obstacle

    # 정적 장애물
    st_obstacle_num = 1
    st_obstacles = [None] * st_obstacle_num
    for i in range(st_obstacle_num):
        st_obstacle = StaticObstacle()

        # 초기 소환 위치 설정
        st_obstacle.set_agent_attribute(px=-2, py=2, vx=0, vy=0, gx=0, gy=0, radius=0.3, v_pref=1, time_step=time_step)
        # 장애물을 사각형으로 설정할 것이라면
        st_obstacle.set_rectangle(width=0.3, height=0.3)

        st_obstacles[i] = st_obstacle

    # 환경에 로봇과 장애물 세팅하기
    env.set_robot(robot)

    # 클래스 담긴 리스트를 넘겨줄지 아니면 클래스 개별로 넘겨줄지는 효율적인 것을 고려해서 수정할 것
    for obstacle in dy_obstacles:
        env.set_dynamic_obstacle(obstacle)
    for obstacle in st_obstacles:
        env.set_static_obstacle(obstacle)

    run_sim(env, episodes=5, render=True)


