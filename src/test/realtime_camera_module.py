import cv2

from src.Modules import *


class Test:
    def run(self, img):
        cv2.imshow('test', cv2.flip(img, 1))


with RealTimeCamera('테스트', 0) as cam:
    cam.run(Test())
