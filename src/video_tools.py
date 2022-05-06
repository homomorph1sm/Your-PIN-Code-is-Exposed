# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt


def extract_frames(video_path, dst_folder, eps):
    video_name = video_path.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video: " + video_path)
        return

    fps = video.get(cv2.CAP_PROP_FPS) 
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    frequency = int(fps / eps)  
    print("EXTRACT_FREQUENCY : {0}".format(frequency))

    count = 1
    index = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % frequency == 0:
            save_path = "{}/{}_{:>03d}.jpg".format(dst_folder, video_name, index)
            cv2.imwrite(save_path, frame)
            index += 1
        count += 1
    video.release()
    print("Totally save {:d} pics".format(index - 1))  


def print_v(v_list):
    x = range(len(v_list))
    v = [v / 30 if v < 500 else 500 / 30 for v in v_list]
    a = [v[i] - v[i - 1] for i in range(1, len(v))]
    a.insert(0, 0)
    # print(v)

    # plt.switch_backend('TkAgg')
    # plt.title("v(pixel per second)")
    # plt.xlabel("x(frame)")
    # plt.ylabel("v(pixel)")
    # plt.plot(x, v)
    # plt.show()

    plt.switch_backend('TkAgg')
    plt.title("a")
    plt.xlabel("x(frame)")
    plt.ylabel("y(pixel)")
    plt.plot(x, a)
    plt.show()


feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def compute_optical_flow(old_img, new_img, old_tag):
    print('using flow compute')

    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    tmp = []
    for i in [-1, -0.5, 0, 0.5, 1]:
        for j in [-1, -0.5, 0, 0.5, 1]:
            tmp.append([[old_tag[0][0] + i, old_tag[0][1] + j]])
    p0 = np.array(tmp, dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    if len(good_new) > 1:
        x, y = good_new.sum(axis=0) / len(good_new)
        print([x, y])
        return [[x, y]]
    else:
        return False


def do_mosaic(frame, x, y, w, h, neighbor=9):

    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            # color = frame[i + y][j + x].tolist()  
            color = [0,0,0]
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  
            cv2.rectangle(frame, left_up, right_down, color, -1)


def do_black(frame, board):
    contours = []
    for node in board:
        contours.append((int(node[0]), int(node[1])))
        contours.append((int(node[0] + node[2]), int(node[1] + node[3])))
    cnt = np.array(contours, dtype=np.float32)
    min_rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(min_rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 0, 0), thickness=cv2.FILLED)
