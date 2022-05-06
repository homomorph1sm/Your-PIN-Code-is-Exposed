import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

import video_tools


def parse_result(dic, threshold=0.4):
    tag_list = []
    n1_list = []
    n3_list = []
    n8_list = []
    for item in dic:
        if item['score'] > threshold:
            if item['category'] == 'tag':
                tag_list.append(item['bbox'])
            elif item['category'] == 'n1':
                n1_list.append(item['bbox'])
            elif item['category'] == 'n3':
                n3_list.append(item['bbox'])
            elif item['category'] == 'n8':
                n8_list.append(item['bbox'])
    return tag_list, n1_list, n3_list, n8_list


class NewSystem:
    video_info = {}
    
    board_not_init = True
    board = [[0, 0, 0, 0] for _ in range(10)]
    fingertip_sequence = [[(0, 0)]]
    last_frame = []
    
    v_list = []
    a_list = [0]
    key_list = []
    result_key_sequence = []
    result_video_path = ''

    def __init__(self, video_path, yolo_result_path='', save_path=''):
        video = cv2.VideoCapture()
        if not video.open(video_path):
            print("can not open the video: " + video_path)
            return
        self.video_info['size'] = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.video_info['fps'] = video.get(cv2.CAP_PROP_FPS)

        if yolo_result_path == '':
            print('real time computing (todo)...')
        else:
            f = open(yolo_result_path, 'r')
            for js in f.readlines():
                ret, frame = video.read()
                if frame is None:
                    print("can not read frame")

                tag_list, n1_list, n3_list, n8_list = parse_result(json.loads(js), threshold=0.1)
                if self.update_board(n1_list, n3_list, n8_list, frame):
                    self.show_status_on_picture(frame, save_path)
                    self.last_frame = frame
            f.close()
        video.release()

    def show_status_on_picture(self, frame, save_path):
        img = frame.copy()
        # add board
        # for i in range(10):
        #     x = int(self.board[i][0] + self.board[i][2] / 2)
        #     y = int(self.board[i][1] + self.board[i][3] / 2)
        #     cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        #     x1 = int(self.board[i][0])
        #     y1 = int(self.board[i][1])
        #     x2 = int(self.board[i][0] + self.board[i][2])
        #     y2 = int(self.board[i][1] + self.board[i][3])
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)
        video_tools.do_black(img, self.board)
        # for i in range(10):
        #     x = int(self.board[i][0])
        #     y = int(self.board[i][1])
        #     w = int(self.board[i][2])
        #     h = int(self.board[i][3])
        #     video_tools.do_mosaic(img, x, y, w, h)

        if save_path:
            if not self.result_video_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.result_video_path = cv2.VideoWriter(save_path, fourcc, self.video_info['fps'], self.video_info['size'])
            self.result_video_path.write(img)
        else:
            cv2.namedWindow('result', 0)
            cv2.resizeWindow("result", 1024, 960);
            cv2.imshow('result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def compute_board(self, n1, n3, n8):
        board = [[0, 0, 0, 0] for _ in range(10)]
        board[1] = n1
        board[3] = n3
        board[8] = n8
        # compute 1,3->2  2,8->5  1,5->9  3,5->7  1,7->4  3,9->6  5,8->0
        board[2] = self.get_mid(board[1], board[3])
        board[5] = self.get_mid(board[2], board[8])
        board[9] = self.get_next(board[1], board[5])
        board[7] = self.get_next(board[3], board[5])
        board[4] = self.get_mid(board[1], board[7])
        board[6] = self.get_mid(board[3], board[9])
        board[0] = self.get_next(board[5], board[8])
        return board

    def update_board(self, n1_list, n3_list, n8_list, frame):
        if n1_list and n3_list and n8_list:
            self.board = self.compute_board(n1_list[0], n3_list[0], n8_list[0])
            self.board_not_init = False
        else:
            if self.board_not_init:
                return False
            else:
                offset = [0] * 4
                if n1_list:
                    tmp = [n1_list[0][i] - self.board[1][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if n3_list:
                    tmp = [n3_list[0][i] - self.board[3][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if n8_list:
                    tmp = [n8_list[0][i] - self.board[8][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if sum(offset) != 0:
                    for i in range(len(self.board)):
                        self.board[i] = [self.board[i][j] + offset[j] for j in range(4)]
        return True

    def get_mid(self, bbox_a, bbox_b):
        bbox = [0] * 4
        for i in range(4):
            bbox[i] = (bbox_a[i] + bbox_b[i]) / 2
        return bbox

    def get_next(self, bbox_a, bbox_b):
        bbox = [0] * 4
        for i in range(4):
            bbox[i] = 2 * bbox_b[i] - bbox_a[i]
        return bbox


class MatchingSystem:
    video_info = {}

    board_not_init = True
    board = [[0, 0, 0, 0] for _ in range(10)]
    fingertip_sequence = [[(0, 0)]]
    last_frame = []

    v_list = []
    a_list = [0]
    key_list = []
    result_key_sequence = []
    result_video_path = ''

    def __init__(self, video_path, yolo_result_path='', save_path=''):
        video = cv2.VideoCapture()
        if not video.open(video_path):
            print("can not open the video: " + video_path)
            return
        self.video_info['size'] = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.video_info['fps'] = video.get(cv2.CAP_PROP_FPS)

        if yolo_result_path == '':
            print('real time computing (todo)...')
        else:
            f = open(yolo_result_path, 'r')
            for js in f.readlines():
                ret, frame = video.read()
                if frame is None:
                    print("can not read frame")

                tag_list, n1_list, n3_list, n8_list = parse_result(json.loads(js), threshold=0.1)
                if self.update_board(n1_list, n3_list, n8_list, frame) and self.update_tag(tag_list, frame):
                    self.compute_key()
                    self.show_status_on_picture(frame, save_path)
                    self.last_frame = frame
            # self.show_all_key()
            f.close()
        video.release()

    def show_status_on_picture(self, frame, save_path):
        img = frame.copy()
        # add board
        for i in range(10):
            x = int(self.board[i][0] + self.board[i][2] / 2)
            y = int(self.board[i][1] + self.board[i][3] / 2)
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            x1 = int(self.board[i][0])
            y1 = int(self.board[i][1])
            x2 = int(self.board[i][0] + self.board[i][2])
            y2 = int(self.board[i][1] + self.board[i][3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)

        # add fingertip, v and a
        fingertip = self.fingertip_sequence[-1]
        cv2.putText(img, '*', (int(fingertip[0][0]), int(fingertip[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.05, (0, 255, 0),
                    5)

        cv2.putText(img, 'v: ' + str(self.v_list[-1]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, 'a: ' + str(self.a_list[-1]), (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # add note
        cv2.putText(img, 'In ' + str(self.key_list[-1]), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        if self.v_list[-1] < 1 and self.key_list[-1]:
            key = self.key_list[-1][0]
            if not self.result_key_sequence or self.result_key_sequence[-1] != key:
                self.result_key_sequence.append(key)
        cv2.putText(img, 'result: ' + str(self.result_key_sequence), (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)

        if save_path:
            if not self.result_video_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.result_video_path = cv2.VideoWriter(save_path, fourcc, self.video_info['fps'],
                                                         self.video_info['size'])
            self.result_video_path.write(img)
        else:
            cv2.namedWindow('result', 0)
            cv2.resizeWindow("result", 1024, 960)
            cv2.imshow('result', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def update_tag(self, tag_list, frame):
        fingertip = []
        if not tag_list:
            # using optical flow compute
            if len(self.fingertip_sequence) > 1:
                fingertip = video_tools.compute_optical_flow(self.last_frame, frame, self.fingertip_sequence[-1])
        else:
            # have YOLO output figertip
            fingertip = self.compute_fingertip(tag_list[0], frame)  # todo
        if fingertip:
            self.fingertip_sequence.append(fingertip)
            return True
        return False

    def compute_board(self, n1, n3, n8):
        board = [[0, 0, 0, 0] for _ in range(10)]
        board[1] = n1
        board[3] = n3
        board[8] = n8
        # compute 1,3->2  2,8->5  1,5->9  3,5->7  1,7->4  3,9->6  5,8->0
        board[2] = self.get_mid(board[1], board[3])
        board[5] = self.get_mid(board[2], board[8])
        board[9] = self.get_next(board[1], board[5])
        board[7] = self.get_next(board[3], board[5])
        board[4] = self.get_mid(board[1], board[7])
        board[6] = self.get_mid(board[3], board[9])
        board[0] = self.get_next(board[5], board[8])
        return board

    def update_board(self, n1_list, n3_list, n8_list, frame):
        if n1_list and n3_list and n8_list:
            self.board = self.compute_board(n1_list[0], n3_list[0], n8_list[0])
            self.board_not_init = False
        else:
            if self.board_not_init:
                return False
            else:
                offset = [0] * 4
                if n1_list:
                    tmp = [n1_list[0][i] - self.board[1][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if n3_list:
                    tmp = [n3_list[0][i] - self.board[3][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if n8_list:
                    tmp = [n8_list[0][i] - self.board[8][i] for i in range(4)]
                    if sum(offset) == 0:
                        offset = tmp
                    else:
                        offset = [(tmp[i] + offset[i]) / 2 for i in range(4)]
                if sum(offset) != 0:
                    for i in range(len(self.board)):
                        self.board[i] = [self.board[i][j] + offset[j] for j in range(4)]
        return True

    def get_mid(self, bbox_a, bbox_b):
        bbox = [0] * 4
        for i in range(4):
            bbox[i] = (bbox_a[i] + bbox_b[i]) / 2
        return bbox

    def get_next(self, bbox_a, bbox_b):
        bbox = [0] * 4
        for i in range(4):
            bbox[i] = 2 * bbox_b[i] - bbox_a[i]
        return bbox

    def compute_fingertip(self, raw_bbox, frame):
        print("using YOLO and ellipse fit")


        result = [[raw_bbox[0] + raw_bbox[2] / 2, raw_bbox[1] + raw_bbox[3] / 2]]
        # try:
        #     
        #     bbox = [int(x) for x in raw_bbox]
        #     img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     ret, thresh = cv2.threshold(img_gray, 100, 255, 0)
        #     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cnt = contours[0]
        #     ellipse = cv2.fitEllipse(cnt)
        #     result = [raw_bbox[0] + ellipse[0][0], raw_bbox[1] + ellipse[0][1]]
        # except:
        #     pass
        print(result)
        return result

    def compute_key(self):
        now = self.fingertip_sequence[-1][0]
        last = self.fingertip_sequence[-2][0]
        v = math.sqrt((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2)
        if self.v_list:
            self.a_list.append(v - self.v_list[-1])
        self.v_list.append(v)

        keys = []
        for i in range(10):
            if self.board[i][0] < now[0] < self.board[i][0] + self.board[i][2] and self.board[i][1] < now[1] < \
                    self.board[i][1] + self.board[i][3]:
                keys.append(i)
                break
        self.key_list.append(keys)

    def show_all_key(self):
        result_key = []
        result_v = []
        for i in range(min(len(self.v_list), len(self.key_sequence))):
            print('-----------')
            print(self.v_list[i])
            print(self.key_sequence[i])
            if sum(self.key_sequence[i]) == 1:
                for j in range(10):
                    if self.key_sequence[i][j] == 1:
                        if not result_key or result_key[-1] != j:
                            result_key.append(j)
                            result_v.append([])
                result_v[-1].append(self.v_list[i])
        print('======')
        print(result_key)
        for l in result_v:
            print(l)
