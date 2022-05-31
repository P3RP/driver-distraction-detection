import numpy as np

from .base import BaseBuffer


class GazeBuffer(BaseBuffer):
    inc_coeff = 0
    dec_coeff = 0
    inc_rate = 0
    dec_rate = 0
    last = {
        'on': 0,
        'off': 0
    }

    def update(self, time, **kwargs):
        pass

    def shift_on(self, shift, frame_time, v, s):
        """
        shift:
        [ 0 ] : gaze != target
        [ 1 ] : 시선이 현재 버퍼로 이동
        [ 2 ] : 버퍼에 시선이 존재
        """
        if shift == 0:
            result = 0
        elif shift == 1:
            result = -1 * (np.log(1 - self.buffer)) / self.inc_coeff
        else:
            result = self.last['on'] + frame_time * (self.weight(v, s) ** -1)

        self.last['on'] = result
        return result

    def shift_off(self, shift, frame_time, v, s):
        """
        shift:
        [ 0 ] : 버퍼에 시선이 존재
        [ 1 ] : 시선이 현재 버퍼에서 벗어남
        [ 2 ] : 버퍼에 시선이 존재하지 않음
        """
        if shift == 0:
            result = 0
        elif shift == 1:
            result = np.log(1 + np.exp(-1 * self.dec_coeff / self.dec_rate) - self.buffer) / self.dec_coeff
            result -= 1 / self.dec_rate
        else:
            result = self.last['off'] + frame_time * self.weight(v, s)

        self.last['off'] = result
        return result
