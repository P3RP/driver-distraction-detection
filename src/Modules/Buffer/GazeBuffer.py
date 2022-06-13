import numpy as np
import time

from .base import BaseBuffer

LATENCY = 0.1


class GazeBuffer(BaseBuffer):
    inc_coeff = 0
    dec_coeff = 0
    inc_rate = 0
    dec_rate = 0
    last = {
        'on': 0,
        'off': 0
    }
    pre_time = time.time()
    duration = 0
    is_gaze = True

    def update(self, f_time, gaze):
        # 초기 설정
        interval = f_time - self.pre_time
        v, s = 100, 100

        # shift 여부 파악
        if self.is_gaze and gaze != self.code:
            self.is_gaze = False
            shift = (0, 1)
        elif not self.is_gaze and gaze == self.code:
            self.is_gaze = True
            shift = (1, 0)
        else:
            shift = (0, 0)

        # Shift Data 설정
        d_off = self.shift_off_v2(gaze == self.code, interval, v, s)
        d_on = self.shift_on_v2(gaze == self.code, interval, v, s)

        # -------------------------------------
        # 버퍼 관리
        # 시선이 영역에 없는 경우
        if gaze != self.code:
            # n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(
            #     self.dec_coeff * (d_shift_off - (1 / self.dec_rate))
            # )
            # n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(self.dec_coeff * (d_shift_off - (1 / self.dec_rate)))
            n_data = self.buffer - np.exp(self.dec_coeff * d_off - (1 / self.dec_rate))
            # n_data = self.buffer -
            self.buffer = max(0, n_data)
        else:
            self.duration += interval
            # 시선 이동 후 대기 시간인 경우
            if self.duration <= LATENCY:
                self.buffer = self.buffer
            # 시선 이동 후 대기 시간 종료
            else:
                # n_data = 1 - np.exp(-1 * self.inc_coeff * d_on)
                n_data = self.buffer + np.exp(-1 * self.inc_coeff * d_on)
                self.buffer = min(1, n_data)

        # -------------------------------------
        # 다음을 위한 환경 설정
        self.pre_time = f_time

    def shift_on(self, is_target, shift, frame_time, v, s):
        if not is_target:
            result = 0
        else:
            if shift[0]:
                result = -1 * np.log1p(0.01 - self.buffer) / self.inc_coeff
            else:
                result = self.last['on'] + frame_time * (self.weight(v, s) ** -1)

        self.last['on'] = result
        return result

    def shift_off(self, is_target, frame_time, v, s):
        if is_target:
            result = 0
        else:
            if self.is_gaze:
                self.duration = 0
                # result = np.log1p(0.01 + np.exp(-1 * self.dec_coeff / self.dec_rate) - self.buffer) / self.dec_coeff
                result = np.log1p(0.01 - self.buffer) / self.dec_coeff
                # result -= 1 / self.dec_rate
            else:
                result = self.last['off'] + frame_time * self.weight(v, s)

        self.last['off'] = result
        return result


    def update_v3(self, frame_time, **kwargs):
        # 초기 설정
        interval = frame_time - self.pre_time
        v, s = 100, 100

        # Shift Data 설정
        d_shift_off = self.shift_off_v2(kwargs['gaze'] == self.code, interval, v, s)
        d_shift_on = self.shift_on_v2(kwargs['gaze'] == self.code, interval, v, s)

        # shift 여부 파악
        if self.is_gaze and kwargs['gaze'] != self.code:
            self.is_gaze = False
        elif not self.is_gaze and kwargs['gaze'] == self.code:
            self.is_gaze = True

        # -------------------------------------
        # 버퍼 관리
        # 시선이 영역에 없는 경우
        if kwargs['gaze'] != self.code:
            # n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(
            #     self.dec_coeff * (d_shift_off - (1 / self.dec_rate))
            # )
            # n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(self.dec_coeff * (d_shift_off - (1 / self.dec_rate)))
            n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(-1 * self.dec_coeff * (d_shift_off - (1 / self.dec_rate)))
            self.buffer = max(0, n_data)
        else:
            self.duration += interval
            # 시선 이동 후 대기 시간인 경우
            if self.duration <= LATENCY:
                self.buffer = self.buffer
            # 시선 이동 후 대기 시간 종료
            else:
                n_data = 1 - np.exp(-1 * self.inc_coeff * d_shift_on)
                self.buffer = min(1, n_data)

        # -------------------------------------
        # 다음을 위한 환경 설정
        self.pre_time = frame_time

    def shift_on_v2(self, is_target, frame_time, v, s):
        if not is_target:
            result = 0
        else:
            result = self.last['on'] + frame_time * (self.weight(v, s) ** -1) * self.inc_rate

        self.last['on'] = result
        return result

    def shift_off_v2(self, is_target, frame_time, v, s):
        if is_target:
            result = 0
        else:
            self.duration = 0
            if self.is_gaze and self.duration > LATENCY:
                result = 0
            else:
                result = self.last['off'] + frame_time * self.weight(v, s)

        self.last['off'] = result
        return result

    def update_v2(self, frame_time, **kwargs):
        # 초기 설정
        interval = frame_time - self.pre_time
        v, s = 100, 100

        # shift 여부 파악
        if self.is_gaze and kwargs['gaze'] != self.code:
            self.is_gaze = False
            self.duration = 0
        elif not self.is_gaze and kwargs['gaze'] == self.code:
            self.is_gaze = True

        # -------------------------------------
        # 버퍼 관리
        # 시선이 영역에 없는 경우
        if kwargs['gaze'] != self.code:
            n_data = self.buffer - interval ** 2
            self.buffer = max(0, n_data)
        else:
            # 영역 보고 있는 지속시간 측정
            self.duration += interval

            # 시선 이동 후 대기 시간인 경우
            if self.duration <= LATENCY:
                self.buffer = self.buffer

            # 시선 이동 후 대기 시간 종료
            else:
                n_data = self.buffer + interval ** 2
                self.buffer = min(2, n_data)

        # -------------------------------------
        # 다음을 위한 환경 설정
        self.pre_time = frame_time

    def update_v4(self, frame_time, **kwargs):
        print(self.last)
        # 초기 설정
        interval = frame_time - self.pre_time
        v, s = 100, 100

        # Shift Data 설정
        d_shift_off = self.shift_off(kwargs['gaze'] == self.code, interval, v, s)
        d_shift_on = self.shift_on(kwargs['gaze'] == self.code, interval, v, s)

        # shift 여부 파악
        if self.is_gaze and kwargs['gaze'] != self.code:
            self.is_gaze = False
        elif not self.is_gaze and kwargs['gaze'] == self.code:
            self.is_gaze = True

        # -------------------------------------
        # 버퍼 관리
        # 시선이 영역에 없는 경우
        if kwargs['gaze'] != self.code:
            n_data = 1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - np.exp(
                self.dec_coeff * (d_shift_off - (1 / self.dec_rate))
            )
            self.buffer = max(0, n_data)
            self.buffer = min(1, self.buffer)

        else:
            self.duration += interval
            # 시선 이동 후 대기 시간인 경우
            if self.duration <= LATENCY:
                self.buffer = self.buffer
            # 시선 이동 후 대기 시간 종료
            else:
                n_data = 1 - np.exp(-1 * self.inc_coeff * d_shift_on)
                self.buffer = min(1, n_data)
                self.buffer = max(0, self.buffer)

        # -------------------------------------
        # 다음을 위한 환경 설정
        self.pre_time = frame_time

    def shift_on_old(self, is_target, frame_time, v, s):
        if not is_target:
            result = 0
        else:
            if not self.is_gaze:
                result = -1 * np.log(1 - self.buffer) / self.inc_coeff
            else:
                result = self.last['on'] + frame_time * (self.weight(v, s) ** -1)

        self.last['on'] = result
        return result

    def shift_off_old(self, is_target, frame_time, v, s):
        if is_target:
            result = 0
        else:
            self.duration = 0
            if self.is_gaze:
                result = np.log(1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - self.buffer) / self.dec_coeff
                result -= 1 / self.dec_rate
            else:
                result = self.last['off'] + frame_time * self.weight(v, s)

        self.last['off'] = result
        return result
