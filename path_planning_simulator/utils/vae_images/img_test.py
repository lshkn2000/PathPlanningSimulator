import os
import cv2
from PIL import Image

path = os.path.abspath(os.getcwd())
files = os.listdir(path)

png_img = []
jpg_img = []
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        png_img.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        jpg_img.append(f)

test_img = png_img[0]
print(type(test_img))
print(test_img.shape) #
