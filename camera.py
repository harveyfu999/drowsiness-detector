import cv2
import math
import numpy as np
from imutils import face_utils
import dlib

import torch
import torch.nn as nn
#import onnx
#import onnxruntime

from model import Net

model = Net()

model.load_state_dict(
    torch.load("model.pt", map_location=torch.device('cpu'))["model_state_dict"])
model.eval()


dummy_input = torch.randn(1, 1, 128, 128, requires_grad=True)
class WrapperModel(nn.Module):
    def __init__(self, real_model):
        super(WrapperModel, self).__init__()
        self.real_model = real_model

    def forward(self, x):
        # model is normalized to 0 to 1
        x = x.transpose(2, 3) # for opencv's NCHW format
        x = self.real_model(x)
        # x = torch.softmax(x, dim=-1)
        return x

wrapperModel = WrapperModel(model)
out = wrapperModel(dummy_input)
torch.onnx.export(wrapperModel,
                  dummy_input,
                  "eye_model.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {
                          0: 'batch_size'
                      },
                      'output': {
                          0: 'batch_size'
                      }
                  })
import os
os.exit()

cam = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "dlib-models/shape_predictor_68_face_landmarks_GTX.dat")
# predictor = dlib.shape_predictor("dlib-models/shape_predictor_5_face_landmarks.dat")
eye_cascade = cv2.CascadeClassifier()
eye_cascade.load("haarcascade_eye_tree_eyeglasses.xml")

def to_rect(r):
    tl = r.tl_corner()
    br = r.br_corner()
    return [(tl.x, tl.y), (br.x, br.y)]


eye_size = 25


def get_leye(shape):
    lx = shape[37][0]
    ly = shape[37][1]
    rx = shape[40][0]
    ry = shape[40][1]
    diff = rx + eye_size - (lx - eye_size)
    return [(lx - eye_size, ly - eye_size),
            (rx + eye_size, ly - eye_size + diff)]


def get_reye(shape):
    lx = shape[43][0]
    ly = shape[43][1]
    rx = shape[46][0]
    ry = shape[46][1]
    diff = rx + eye_size - (lx - eye_size)
    return [(lx - eye_size, ly - eye_size),
            (rx + eye_size, ly - eye_size + diff)]


def crop_from_rect(img, rect):
    return img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]


def run_eye(gray, eye):
    eye_img = crop_from_rect(gray, eye)
    eye_img = cv2.resize(eye_img, dsize=(128, 128))
    eye_img = eye_img[np.newaxis, np.newaxis, ...]

    #eye_img = (2 / 255 * eye_img - 1) * 3
    normalized_img = np.zeros((128, 128))
    normalized_img = cv2.normalize(eye_img, normalized_img, 0, 255, cv2.NORM_MINMAX)
    normalized_img = (2 / 255 * normalized_img - 1)

    with torch.no_grad():
        out = model(torch.tensor(normalized_img, dtype=torch.float32)).flatten()
        out = torch.softmax(out, dim=-1)
    return out, eye_img


frame = 0
old_ear = 1
cur_ear = 1
while True:
    check, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    for eye in eyes:
        x = eye[0]
        y = eye[1]
        w = eye[2]
        h = eye[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    if rects:
        rect = to_rect(rects[0])
        cv2.rectangle(img, rect[0], rect[1], (255, 0, 0), 2)

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        #for s in shape:
        #    img = cv2.circle(img, [s[0], s[1]], 2, (0, 255, 0))
        #img = cv2.circle(img, [shape[38][0], shape[38][1]], 2, (0, 255, 255))
        #img = cv2.circle(img, [shape[41][0], shape[41][1]], 2, (0, 255, 255))
        #img = cv2.circle(img, [shape[36][0], shape[36][1]], 2, (0, 255, 255))
        #img = cv2.circle(img, [shape[39][0], shape[39][1]], 2, (0, 255, 255))
        #old_ear = cur_ear
        #cur_ear = float(shape[38][1] - shape[41][1]) / float(shape[36][0] - shape[39][0])
        #print(cur_ear)
        if shape.size > 0:
            gray = cv2.GaussianBlur(gray, (3, 3), 0.4)

            leye = get_leye(shape)
            reye = get_reye(shape)

            cv2.rectangle(img, leye[0], leye[1], (255, 0, 0), 2)
            cv2.rectangle(img, reye[0], reye[1], (255, 0, 0), 2)

            leye_out, leye_img = run_eye(gray, leye)
            reye_out, reye_img = run_eye(gray, reye)

            leye_open = leye_out.numpy()[0] < 0.98
            reye_open = reye_out.numpy()[0] < 0.98

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

        face = crop_from_rect(img, rect)
        #cv2.imshow("face", face)
        cv2.imshow("leye", leye_img[0,0])
        cv2.imshow("reye", reye_img[0,0])

    cv2.imshow("img", img)
    frame += 1

    key = cv2.waitKey(5)
    if key == 115:
        if leye_img is not None:
            from datetime import datetime
            cv2.imwrite(
                f"own-images/open/leye-{datetime.now().strftime('%H-%M-%S')}.png",
                leye_img)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
