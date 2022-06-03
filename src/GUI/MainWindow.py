import time
import json
import random

import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg

from src.utils.dirutil import get_root_path
from src.Modules import FrameSource


# CONFIG
config = json.load(open(get_root_path() + '/GUI/config/config.json', 'r'))      # 상황별 색 설정
ui_src = get_root_path() + '/GUI/config/main.ui'    # UI source 경로


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cam_f: FrameSource, cam_s: FrameSource, interval: int, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('Driver Distraction Detection')

        # ------------------------------------------------------
        # 필요한 변수 설정
        self.s_time = None          # 시작 시간 관리
        self.timer = None           # 작동 Frame 관리
        self.interval = interval    # Frame 간격 관리

        # Qt Designer에서 생성한 UI 파일 불러오기
        uic.loadUi(ui_src, self)

        # ------------------------------------------------------
        # GUI 영역 세팅
        self.start_btn.clicked.connect(self.start)  # 시작 버튼에 start 함수 연동
        self.stop_btn.clicked.connect(self.stop)    # 중지 버튼에 stop 함수 연동

        # ------------------------------------------------------
        # 카메라 세팅
        self.cam_f = cam_f  # 정면 카메라 세팅
        self.cam_s = cam_s   # 측면 카메라 세팅

        # ------------------------------------------------------
        # 그래프 설정
        # 각 그래프 Y축 영역 설정 (0 ~ 1, 패딩 0.1)
        self.o_graph.setYRange(0, 1, padding=0.1)
        self.n_graph.setYRange(0, 1, padding=0.1)

        # 기준 선 그리기
        self.o_graph.addLine(x=None, y=0.2, pen=pg.mkPen('r', width=2))
        self.n_graph.addLine(x=None, y=0.2, pen=pg.mkPen('r', width=2))

        # 각 그래프 데이터 변수 설정
        self.o_chart = self.o_graph.plot()  # 기존 알고리즘 차트
        self.n_chart = self.n_graph.plot()  # 개선 알고리즘 차트
        self.o_x = []   # 기존 알고리즘 X 값 (시간 값)
        self.o_y = []   # 기존 알고리즘 Y 값 (Buffer 값)
        self.n_x = []   # 개선 알고리즘 X 값 (시간 값)
        self.n_y = []   # 개선 알고리즘 Y 값 (Buffer 값)

    def start(self):
        """
        시작 버튼을 눌렀을 때 작동하는 함수
        """
        # 시작 시간 설정
        self.s_time = time.time()

        # 카메라 작동 시작
        self.cam_f.start()
        self.cam_s.start()

        # 반복을 위한 Timer 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.interval)    # FPS 관리 (MicroSecond)

    def stop(self):
        """
        종료 버튼을 눌렀을 때 작동하는 함수
        """
        # 반복을 위한 Timer 종료
        self.timer.stop()

        # 카메라 작동 종료
        self.cam_f.stop()
        self.cam_s.stop()

        # 그래프 데이터 초기화
        self.o_x = []
        self.o_y = []
        self.n_x = []
        self.n_y = []

    def update_chart(self):
        """
        각 차트의 데이터를 업데이트하는 함수
        """
        self.n_chart.setData(x=self.n_x[-100:], y=self.n_y[-100:])
        self.o_chart.setData(x=self.o_x[-100:], y=self.o_y[-100:])

    def display_image(self, img: QImage, window):
        """
        특정 Label에 이미지를 작성하기 위한 함수
        """

        display_label: QtWidgets.QLabel = getattr(self, window, None)
        if display_label is None:
            raise ValueError(f"No such display window in GUI: {window}")

        display_label.setPixmap(QPixmap.fromImage(img))
        display_label.setScaledContents(True)

    def display_label(self, window, text, style):
        """
        특정 Label에 이미지를 작성하기 위한 함수
        """

        display_label: QtWidgets.QLabel = getattr(self, window, None)
        if display_label is None:
            raise ValueError(f"No such display window in GUI: {window}")

        display_label.setText(text)
        display_label.setStyleSheet(style)

    def update_frame(self):
        """
        매 프레임마다 GUI 상황을 업데이트해주는 함수
        """
        # ------------------------------------------------------
        # 카메라 화면 설정
        # 정면 카메라 화면 설정
        front_frame = self.cam_f.next_frame()
        self.display_image(self.opencv_to_qt(front_frame), window="front_cam")

        # 측면 카메라 화면 설정
        side_frame = self.cam_s.next_frame()
        self.display_image(self.opencv_to_qt(side_frame), window="side_cam")

        # ------------------------------------------------------
        # TODO: 운전 상황에 대한 Label 설정
        env = 0     # 운전 상황 가져오기

        # 운전 상황에 맞는 설정 불러오기
        if env == 0:
            driving_case = config['driving']['forward']
        elif env == 1:
            driving_case = config['driving']['backward']
        else:
            driving_case = config['driving']['forward']

        # Label 수정
        self.display_label("driving", driving_case['msg'], driving_case['style'])

        # ------------------------------------------------------
        now = time.time() - self.s_time

        # 측면 사진을 통해 행동 추론 및 라벨 수정
        # TODO: 행동 추론
        activity = int(now) % 10

        # 행동 라벨 수정
        act_label = config['activity'][str(activity)]
        self.display_label("activity", act_label['msg'], act_label['style'])

        # ------------------------------------------------------
        # 정면 사진을 통해 시선 영역 추록 및 라벨 수정
        # TODO: 시선 영역 추록
        gaze_zone = int(now) % 7

        # 시선 라벨 수정
        gaze_label = config['gaze'][str(gaze_zone)]
        self.display_label("gaze", gaze_label['msg'], gaze_label['style'])
        # self.display_label("activity", str(random.uniform(0.0, 1.0)), "background-color: green;")

        # ------------------------------------------------------
        # TODO: 행동과 시선을 통해 버퍼 데이터 업데이트

        # ------------------------------------------------------
        # 각 차트에 데이터 넣기 및 라벨 설정
        now = time.time() - self.s_time

        # 1. AttenD Graph
        self.o_x.append(now)
        o_data = random.uniform(0.0, 1.0)       # TODO : AttenD 버퍼 데이터 넣기
        self.o_y.append(o_data)
        # 라벨 설정
        if o_data < 0.2:
            o_case = config['status']['distract']
        else:
            o_case = config['status']['safe']
        self.display_label("o_status", o_case['msg'], o_case['style'])

        # 2. 개선 Graph
        self.n_x.append(now)
        n_data = random.uniform(0.0, 1.0)       # TODO : 개선 버퍼 데이터 넣기
        self.n_y.append(n_data)
        # 라벨 설정
        if n_data < 0.2:
            n_case = config['status']['distract']
        else:
            n_case = config['status']['safe']
        self.display_label("n_status", n_case['msg'], n_case['style'])

        # 차트 업데이트
        self.update_chart()

    @staticmethod
    def opencv_to_qt(img) -> QImage:
        """
        OpenCV를 통해서 읽어들인 이미지를 PyQt 이미지로 수정해주는 함수
        """
        qformat = QImage.Format.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                qformat = QImage.Format.Format_RGBA8888
            else:  # RGB
                qformat = QImage.Format.Format_RGB888

        img = np.require(img, np.uint8, "C")
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)  # BGR to RGB
        out_image = out_image.rgbSwapped()

        return out_image
