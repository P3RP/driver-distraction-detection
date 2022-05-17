import cv2


class RealTimeCamera:
    name: str = None
    cam_id: int = -1
    cam = None

    def __init__(self, name: str, cam_id: int):
        self.name = name
        self.cam_id = cam_id

    def run(self, module, **kwargs):
        """
        실시간 영상을 카메라를 통해 가져오며 입력받은 모듈을 통해서 처리하는 함수
        :param module: 작업을 수행할 Module
        :return:
        """
        while self.cam.isOpened():
            success, image = self.cam.read()

            if not success:
                print(f'{self.name} 카메라 검색 실패')
                continue

            # 실시간 영상으로 작업하려는 모듈을 객체로 넘겨야 함
            # 실행할 작업을 run() 함수로 구현해야 함
            module.run(image, **kwargs)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    def __enter__(self):
        """
        with문을 통해서 Camera 사용
        :return:
        """
        self.cam = cv2.VideoCapture(self.cam_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with문 탈출 시 cam release
        :return:
        """
        self.cam.release()
