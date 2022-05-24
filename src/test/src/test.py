from matplotlib import pyplot as plt

from src.buffer import *


def gaze(now):
    if 1.3 <= now < 2.3:
        return "-1"

    if 6 <= now < 7.3:
        return "-1"

    if 3 <= now < 3.5:
        return "1"

    if 10 <= now < 10.3:
        return "2"

    if 11 <= now < 11.4:
        return "3"

    if 20 <= now < 21:
        return "4"

    if 21 <= now < 21.3:
        return "-1"

    if 21.3 <= now < 22:
        return "6"

    if 22 <= now < 23:
        return "2"

    if 15 <= now:
        return "7"
    return "0"


def env(now):
    if now > 15:
        return "1"
    return "0"


def activity(now):
    if 4 <= now < 6:
        return "1"

    if 7 <= now < 8.5:
        return "1"

    if 20 <= now < 23:
        return "0"

    if 15 <= now:
        return "2"
    return "0"


if __name__ == "__main__":
    time = 0
    step = 0.1

    buffers = [
        ForwardBuffer(),
        DashBoardBuffer(),
        RearViewMirrorBuffer(),
        WingMirrorLBuffer(),
        WingMirrorRBuffer()
    ]
    # back =
    k = [0.8, 0.05, 0.05, 0.05, 0.05]

    ac = ActivityBuffer()
    result = [
        [],
        [],
        [],
        [],
        [],
    ]
    times = []
    main = []
    act = []
    up = []
    back = []

    # ========================================
    a = True
    # ========================================

    goback = False

    while time < 30:
        if a:
            if time >= 15 and not goback:
                result.append([])
                result.append([])
                buffers.append(BackBuffer())
                buffers.append(BackCameraBuffer())
                goback = True

            if 21.3 <= time <= 23.1:
                k = [-0.3, 0.05, 0.25, 0.25, 0.25, 0.0, 0.25]
            elif time >= 15:
                k = [-0.3, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05]
            else:
                k = [0.8, 0.05, 0.05, 0.05, 0.05]

        print(time)
        print(gaze(time))
        print(activity(time))
        if time >= 15:
            back.append(time)

        times.append(time)
        t = 0

        for i, buffer in enumerate(buffers):
            buffer.update(gaze(time), 0.1)
            print(buffer.name)
            print(buffer.buffer)
            result[i].append(buffer.buffer)
            if buffer.buffer:
                t += buffer.buffer * k[i]
        main.append(t)
        print("기존 : ", t)
        time += step

        ac.update(env(time), gaze(time), activity(time), 0.1)
        act.append(ac.buffer)
        print("act", ac.buffer)

        print("UP : ", t*ac.buffer)
        up.append(min(1, t * ac.buffer))

        print("===============================")
    color = [
        "#66FF66",
        "#FF9999",
        "#FF66FF",
        "#FFFF66",
        "#FFB366",
        "#00FFFF",
        "#0000CC",
    ]

    plt.figure(figsize=(100, 8))
    plt.gca().spines['right'].set_visible(False)  # 오른쪽 테두리 제거
    plt.gca().spines['top'].set_visible(False)  # 위 테두리 제거
    plt.gca().spines['bottom'].set_visible(False)  # 위 테두리 제거
    plt.gca().spines['left'].set_visible(False)  # 위 테두리 제거
    plt.yticks(ticks=[])  # y축 tick 제거
    plt.xticks(ticks=[])  # y축 tick 제거
    # col = 1
    # # plt.plot(times, result[0])
    # # for i in range(5):
    # #     plt.plot(times, result[i], color=color[i], linewidth='10')
    # # plt.plot(back, result[5], color=color[5], linewidth='10')
    # # plt.plot(back, result[6], color=color[6], linewidth='10')
    # plt.plot(times, act, linewidth='10', color="#000000")
    # plt.plot(times, main, linewidth='10', color="#000000")
    plt.plot(times, up, linewidth='10', color="#000000")
    # # plt.plot(back, result[5], linewidth='10')
    plt.savefig('myfigure.png', transparent=True)
    plt.show()

