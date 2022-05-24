from .base import BaseBuffer
import math


class ForwardBuffer(BaseBuffer):
    name = 'Forward'
    code = '0'
    type = 0
    down_p = 1.5
    up_p = 1.5

    def down(self, gaze, time):
        add = 1.5
        if gaze != "-1":
            add = 0.6

        self.buffer -= math.exp(-1 * self.buffer) * time * add
        self.last_glance = False
        self.delay = 0
