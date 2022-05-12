from .base import BaseBuffer
import math


class ActivityBuffer:
    type = 1
    buffer = 1

    act = {
        "0": {
            "0": 1,
            "1": -0.8,
            "2": -1.0
        },
        "1": {
            "0": -1,
            "1": -1,
            "2": 1
        }
    }

    def update(self, env, gaze, activity, time):
        if env == "1":
            if activity == "0":
                if gaze in ["2", "3", "4", "6"]:
                    self.buffer += 1 * time * math.exp(-0.1 * self.buffer) * math.exp(
                        -1 * self.buffer)
                    if self.buffer >= 1:
                        self.buffer = 1

                    if self.buffer < 0:
                        self.buffer = 0
                    return

        self.buffer += self.act[env][activity] * time * math.exp(-0.1 * self.buffer) * math.exp(-1 * self.buffer)
        if self.buffer >= 1:
            self.buffer = 1

        if self.buffer < 0:
            self.buffer = 0
