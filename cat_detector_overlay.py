import cv2
import numpy as np
import mss
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time

KERNELSIZE = 608

def load_yolo(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

net_v2, out_v2 = load_yolo('yolov2.cfg', 'yolov2.weights')
net_v3, out_v3 = load_yolo('yolov3.cfg', 'yolov3.weights')

CAT_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.5

sct = mss.mss()
monitor = sct.monitors[1]

def resize_with_padding(frame, target_size=(KERNELSIZE, KERNELSIZE)):
    h, w, _ = frame.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    padded = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return padded, scale, pad_x, pad_y

def correct_boxes(boxes, scale, pad_x, pad_y, original_width, original_height):
    corrected = []
    for (x, y, w, h) in boxes:
        x = (x - pad_x) / scale
        y = (y - pad_y) / scale
        w = w / scale
        h = h / scale
        x = max(0, min(x, original_width - 1))
        y = max(0, min(y, original_height - 1))
        w = max(0, min(w, original_width - x))
        h = max(0, min(h, original_height - y))
        corrected.append((int(x), int(y), int(w), int(h)))
    return corrected

class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(QtWidgets.QApplication.desktop().screenGeometry())
        self.detections = []  # list of (x, y, w, h, color)
        self.show()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        for (x, y, w, h, color) in self.detections:
            painter.setPen(QtGui.QPen(QtGui.QColor(*color), 3))
            painter.drawRect(x, y, w, h)

def detect_cats(net, out_layers, frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (KERNELSIZE, KERNELSIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == CAT_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                center_x, center_y, width, height = detection[0:4]
                x = int((center_x - width/2) * KERNELSIZE)
                y = int((center_y - height/2) * KERNELSIZE)
                w_box = int(width * KERNELSIZE)
                h_box = int(height * KERNELSIZE)
                boxes.append((x, y, w_box, h_box))
    return boxes

# Main app
app = QtWidgets.QApplication(sys.argv)
overlay = Overlay()

while True:
    start_time = time.time()

    screen = np.array(sct.grab(monitor))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

    original_height, original_width = screen.shape[:2]
    padded_frame, scale, pad_x, pad_y = resize_with_padding(screen, (KERNELSIZE, KERNELSIZE))

    # Get YOLOv2 (red) and YOLOv3 (green) cats
    cats_v2 = detect_cats(net_v2, out_v2, padded_frame)
    cats_v3 = detect_cats(net_v3, out_v3, padded_frame)

    cats_v2 = correct_boxes(cats_v2, scale, pad_x, pad_y, original_width, original_height)
    cats_v3 = correct_boxes(cats_v3, scale, pad_x, pad_y, original_width, original_height)

    detections = []

    for box in cats_v2:
        detections.append((box[0], box[1], box[2], box[3], (255, 0, 0)))  # red = YOLOv2

    for box in cats_v3:
        detections.append((box[0], box[1], box[2], box[3], (0, 255, 0)))  # green = YOLOv3

    overlay.detections = detections
    overlay.update()
    QtWidgets.QApplication.processEvents()

    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}", end="\r")
