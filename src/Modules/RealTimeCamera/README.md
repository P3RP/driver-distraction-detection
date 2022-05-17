# RealTimeCamera

OpenCV를 이용하여 카메라에서 실시간으로 영상을 가져오는 모듈

## 사용법

```python
import cv2

from src.Modules import *


# 실시간 영상을 통해서 작업을 수행할 클래스 
class Test:
    # RealTimeCamera의 run() 함수 내부에서 해당 클래스의 run을 실행
    # 무조건 존재해야 함
    def run(self, img):
        cv2.imshow('test', cv2.flip(img, 1))

# 아래와 같이 작성
with RealTimeCamera(name='테스트', cam_id=0) as cam:
    cam.run(Test())

```