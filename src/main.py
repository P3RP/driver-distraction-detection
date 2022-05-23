import cv2

from src.Modules import *

# ---------------------------------------------
# 1. 초기 설정
# 1.1. 측면 카메라 생성
cam_side = cv2.VideoCapture(0)
# 1.2. 정면 카메라 생성
cam_front = cv2.VideoCapture(1)

# TODO: 1.3. 행동 분류 모델 로딩

# TODO: 1.4. 시선 분류 모델 로딩

# TODO: 1.5. 행동 버퍼 생성

# TODO: 1.6. 시선 버퍼 생성

# ---------------------------------------------
# 2. 카메라에 대한 각 작업
while cam_side.isOpened() or cam_front.isOpened():
    # 2.1. 측면 카메라 프레임 불러오기
    success_side, image_side = cam_side.read()
    if not success_side:
        print('측면 카메라 검색 실패')
        continue

    # 2.2. 측면 카메라 프레임 불러오기
    success_front, image_front = cam_front.read()
    if not success_front:
        print('정면 카메라 검색 실패')
        continue

    # TODO: 2.3. 측면 이미지를 통한 행동 분류

    # TODO: 2.4. 행동 벡터 값 갱신

    # TODO: 2.5. 측면 이미지를 통한 머리 벡터 인식

    # TODO: 2.6. 정면 이미지 + 머리 벡터(측면)을 통한 시선 영역 분류

    # TODO: 2.7. 시선 버퍼 갱신

    # 2.x. 종료 조건 설정
    if cv2.waitKey(5) & 0xFF == 27:
        break

# ---------------------------------------------
# 3. 종료
# 3.1. 카메라 종료
cam_side.release()
cam_front.release()



