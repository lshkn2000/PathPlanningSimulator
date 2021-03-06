import os
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

"""
World Model 학습
실행 순서는 make_img -> make_img_dataset -> train_cnn_vae
시뮬레이션의 정보를 바탕으로 이미지 만드는 파일이다. 
"""

# color setting
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
white_color = (255, 255, 255)

map_size = (10, 10)
pixel_size = (126, 126)
img_frame = (pixel_size[0], pixel_size[1], 3)


def make_directories(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class TransformMeter2Pixel():
    def __init__(self, map_size, pixel_size):
        """
        :param map_size: [width, height]
        :param img_frame: [width, height]
        """
        self.map_size = map_size
        self.pixel_size = pixel_size

        self.scale_width = pixel_size[0] / map_size[0]
        self.scale_height = pixel_size[1] / map_size[1]
        self.scale_mean = (self.scale_width + self.scale_height) / 2

        print(f"scale width : {self.scale_width} \n scale height : {self.scale_height} \n scale pixel radius : {self.scale_mean}")

    def transform(self, *args):
        """
        :param args: [px py radius]
        :return:
        """
        pixel_width = int(self.scale_width * args[0] + self.pixel_size[0] / 2)
        pixel_height = int(self.pixel_size[1] - ((self.scale_height * args[1]) + (self.pixel_size[1] / 2)))
        pixel_width = max(min(pixel_width, self.pixel_size[0]), 0)
        pixel_height = max(min(pixel_height, self.pixel_size[1]), 0)
        pixel_obstacle_position = (pixel_width, pixel_height)
        pixel_obstacle_size = int(self.scale_mean * args[2])
        return pixel_obstacle_position, pixel_obstacle_size


def main():
    transformer = TransformMeter2Pixel(map_size, pixel_size)

    # RVO2로 얻은 dataset 가져오기
    PATH = r'/home/rvlab/PathPlanningSimulator_branch/PathPlanningSimulator_new_worldcoord_2/path_planning_simulator'
    PRETRAIN_BUFFER_PATH = os.path.join(PATH,'vae_ckpts/simulation_buffer_dict.pkl')

    # 이미지 데이터셋 저장할 장소 지정
    IMG_DATASET_PATH = os.path.join(PATH, 'vae_ckpts/img_dataset')
    make_directories(IMG_DATASET_PATH)

    if os.path.isfile(PRETRAIN_BUFFER_PATH):
        with open(PRETRAIN_BUFFER_PATH, 'rb') as f:
            buffer_dict = pickle.load(f)
            data_list = buffer_dict["pretrain"]

            print("Start make img dataset!")
            for idx, data in enumerate(tqdm(data_list, desc="Make Img Dataset From PreTrain Data")):
                state = data[0]
                robot_info = state[:7]
                obstacles_info = state[7:]

                robot_position_n_radius = [robot_info[0], robot_info[1], robot_info[-1]]
                robot_pixel_position, robot_pixel_radius = transformer.transform(*robot_position_n_radius)
                robot_goal_position = (robot_info[4], robot_info[5], 0.3) # 0.3 (m) 은 goal 인정 범위
                goal_pixel_position, goal_pixel_radius = transformer.transform(*robot_goal_position)

                # 도화지 준비
                img = np.zeros(img_frame, np.uint8)
                cv2.circle(img, robot_pixel_position, robot_pixel_radius, green_color, -1)
                cv2.circle(img, goal_pixel_position, goal_pixel_radius, red_color, -1)

                for i in range(int(len(obstacles_info) / 5)):
                    obstacle_position_n_radius = (obstacles_info[5 * i], obstacles_info[5 * i + 1], obstacles_info[5 * i + 4])
                    obstacle_pixel_position, obstacle_pixel_radius = transformer.transform(*obstacle_position_n_radius)
                    cv2.circle(img, obstacle_pixel_position, obstacle_pixel_radius, blue_color, -1)

                # cv2.imshow("test", img)
                # cv2.waitKey(0)
                img_name = IMG_DATASET_PATH + r"/" + str(idx) + ".jpg"
                cv2.imwrite(img_name, img)


if __name__ == '__main__':
    main()