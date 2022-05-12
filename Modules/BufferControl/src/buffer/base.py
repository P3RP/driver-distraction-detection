import math


class BaseBuffer:
    name = "base"
    code = "-1"
    down_p = 1
    up_p = 1
    type = -1
    delay = 0
    buffer = 1
    last_glance = True

    def update(self, gaze, time):
        # 시선이 해당 영역에 있는 경우
        if gaze == self.code:
            # 이전 시선 역시 이 Buffer인 경우
            if self.last_glance:
                # Buffer가 최대치인 경우
                if self.buffer >= 1:
                    self.stay()
                # 최대치가 아닌 경우
                else:
                    self.up(time)
            # 이전 시선이 Buffer가 아닌 경우
            else:
                # Delay가 0.1초가 되었는지 확인
                if self.delay >= 0.1:
                    self.last_glance = True
                    # Buffer가 최대인 경우
                    if self.buffer >= 1:
                        self.stay()
                    else:
                        self.up(time)
                # Delay가 0.1초가 안 되는 경우
                else:
                    self.delay += time
                    self.stay()
        # 시선이 해당 영역에 없는 경우
        else:
            self.down(gaze, time)
            if self.buffer < 0:
                self.buffer = 0

    def down(self, gaze, time):
        self.buffer -= math.exp(-1 * self.buffer) * time * self.down_p
        self.last_glance = False
        self.delay = 0

    def up(self, time):
        self.buffer += math.exp(-1 * self.buffer) * time * self.up_p
        if self.buffer >= 1:
            self.buffer = 1
        self.last_glance = True

    def stay(self):
        self.last_glance = True
