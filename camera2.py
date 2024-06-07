import cv2
import math
import numpy as np
from imutils import face_utils
import dlib

import numpy as np

import torch
import torch.nn as nn
import onnx
import onnxruntime as rt

import matplotlib.pyplot as plt

from model import Net
import box_utils

model = Net()

model.load_state_dict(
    torch.load("model.pt", map_location=torch.device('cpu'))["model_state_dict"])
model.eval()

sess_face_detection = rt.InferenceSession("3rdparty-models/version-RFB-320-int8.onnx")
face_detection_input_name = sess_face_detection.get_inputs()[0].name

sess_landmark_model = rt.InferenceSession("3rdparty-models/2d106det.onnx")
landmark_model_input_name = sess_landmark_model.get_inputs()[0].name

cam = cv2.VideoCapture(0)

# not that img should be in the form (height, width)
def scale_pyramid(img, scale_factor=1.3, n = 3):
    h, w = img.shape
    w = float(w)
    h = float(h)
    center_x = w/2.0
    center_y = h/2.0

    ret = []
    for i in range(n):
        scaled_w_over_2 = w / pow(scale_factor, i) / 2.0
        scaled_h_over_2 = h / pow(scale_factor, i) / 2.0

        x1 = int(center_x - scaled_w_over_2)
        x2 = int(center_x + scaled_w_over_2)

        y1 = int(center_y - scaled_w_over_2)
        y2 = int(center_y + scaled_w_over_2)
        if not (y2 - y1 > 2 and x2 - x1 > 2):
            break
        ret.append(cv2.resize(img[y1:y2, x1:x2], dsize=(128, 128)))

    return np.stack(ret)

def run_eye(gray, eye, which_eye="left"):
    eye_img = gray[eye[1]:eye[3], eye[0]:eye[2]].copy()
    eye_img = cv2.resize(eye_img, dsize=(128, 128))
    #eye_img = eye_img[np.newaxis, np.newaxis, ...]
    imgs = scale_pyramid(eye_img)
    if which_eye == "left":
        cv2.imshow("0", imgs[0])
        cv2.imshow("1", imgs[1])
        cv2.imshow("2", imgs[-1])

    #eye_img = (2 / 255 * eye_img - 1) * 3
    #normalized_img = np.zeros((128, 128))
    #normalized_img = cv2.normalize(eye_img, normalized_img, 0, 255, cv2.NORM_MINMAX)
    #normalized_img = (2 / 255 * normalized_img - 1)

    imgs = (2 / 255 * imgs - 1)
    #imgs = np.transpose(imgs, [0, 2, 1])
    imgs = np.expand_dims(imgs, axis=1)

    with torch.no_grad():
        out = model(torch.tensor(imgs, dtype=torch.float32))
        out = torch.softmax(out, dim=-1)
        out = out.transpose(0, 1)
        if which_eye == "left":
            print(out[0], out[1])
            plt.clf()
            plt.ylim(0, 1)
            plt.plot(out[0])
        out = torch.mean(out, dim=1)
    return out, gray[eye[1]:eye[3], eye[0]:eye[2]]

eye_size = 50
def get_leye(shape):
    lx = shape[35][0]
    ly = shape[35][1]
    rx = shape[39][0]
    ry = shape[39][1]
    diff = max(rx - lx, eye_size) //2
    center_x = (lx + rx)//2
    center_y = (ly + ry)//2
    return [center_x - diff, center_y - diff, center_x + diff, center_y + diff]


def get_reye(shape):
    lx = shape[89][0]
    ly = shape[89][1]
    rx = shape[93][0]
    ry = shape[93][1]
    diff = max(rx - lx, eye_size) //2
    center_x = (lx + rx)//2
    center_y = (ly + ry)//2
    return [center_x - diff, center_y - diff, center_x + diff, center_y + diff]

while True:
    check, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # channels x height x width ---> x and y coordinate
    # normalized to 0 to 1
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(rgb_img, (320, 240)).astype(np.float32)
    input_img = (input_img - np.array([127, 127, 127])) / 128.0
    input_img = np.transpose(input_img, [2, 0, 1])
    input_img = np.expand_dims(input_img, axis=0)
    output = sess_face_detection.run(None, {face_detection_input_name: input_img.astype(np.float32)})
    boxes, _, _ = box_utils.predict(img.shape[1], img.shape[0], output[0], output[1], 0.8)

    for b in boxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

        input_img = rgb_img[b[1]:b[3], b[0]:b[2], :]
        input_img = cv2.resize(input_img, (192, 192)).astype(np.float32)
        #input_img = (input_img - np.array([127, 127, 127])) / 128.0
        # 0 -255
        input_img = np.transpose(input_img, [2, 0, 1]) # channel x height x width
        input_img = np.expand_dims(input_img, axis=0)
        output, = sess_landmark_model.run(None, {landmark_model_input_name: input_img.astype(np.float32)})

        shape = np.zeros((106, 2))
        for i in range(106):
            shape[i][0] = (output[0][i*2] + 1) / 2 * (b[2] - b[0]) # y coordinate
            shape[i][1] = (output[0][i*2 + 1] + 1) / 2 * (b[3] - b[1]) # x coordinate
        shape = shape.astype(np.int32)
        leye_rect = get_leye(shape)
        reye_rect = get_reye(shape)
        for i in range(106):
            cv2.circle(img, (int(b[0] + shape[i][0]), int(b[1] + shape[i][1])), 1, (255, 0, 0), 2, cv2.LINE_AA)

        #cv2.circle(img, (b[0] + shape[35][0], b[1] + shape[35][1]), 1, (255, 0, 0), 2, cv2.LINE_AA)
        #cv2.circle(img, (b[0] + shape[93][0], b[1] + shape[93][1]), 1, (255, 0, 0), 2, cv2.LINE_AA)

        #cv2.rectangle(img, (leye_rect[1], leye_rect[3]), (leye_rect[0], leye_rect[2]), (255, 0, 0), 2)
        cv2.rectangle(img, (int(b[0] + leye_rect[0]), int(b[1] + leye_rect[1])), (int(b[0] + leye_rect[2]), int(b[1] + leye_rect[3])), (255, 0, 0), 2)
        cv2.rectangle(img, (int(b[0] + reye_rect[0]), int(b[1] + reye_rect[1])), (int(b[0] + reye_rect[2]), int(b[1] + reye_rect[3])), (255, 0, 0), 2)

        leye_out, leye_img = run_eye(gray[b[1]:b[3], b[0]:b[2]], leye_rect, which_eye="left")
        reye_out, reye_img = run_eye(gray[b[1]:b[3], b[0]:b[2]], reye_rect, which_eye="right")
        leye_open = leye_out.numpy()[0] < 0.80
        reye_open = reye_out.numpy()[0] < 0.80
        cv2.imshow("leye_img", leye_img)

        img = cv2.putText(img,
                            f"0 - closed, 100 - open", (100, 50),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            color=(0, 255, 255),
                            thickness=7)
        img = cv2.putText(img,
                            f"right: {leye_out.numpy()[0]*100:.2f} {leye_out.numpy()[1]*100:.2f}", (100, 100),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            color=(100, 0, 255) if leye_open else (0, 255, 0),
                            thickness=5)
        img = cv2.putText(img,
                            f"left: {reye_out.numpy()[0]*100:.2f} {reye_out.numpy()[1]*100:.2f}", (100, 200),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            color=(100, 0, 255) if reye_open else (0, 255, 0),
                            thickness=5)

        break

    cv2.imshow("img", img)
    plt.pause(0.05)

    key = cv2.waitKey(100)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
