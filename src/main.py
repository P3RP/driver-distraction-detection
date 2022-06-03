import sys

from PyQt5 import QtWidgets

from GUI import MainWindow
from Modules import FrameSource


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow(
        cam_f=FrameSource('../data/front.mp4'),
        cam_s=FrameSource('../data/side.mp4'),
        interval=33
    )
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
