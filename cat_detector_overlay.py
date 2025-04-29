import cv2
import numpy as np
import mss
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time

# Load YOLOv2 and YOLOv3
def load_yolo(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

net_v2, out_v2 = load_yolo('yolov2.cfg', 'yolov2.weights')
net_v3, out_v3 = load_yolo('yolov3.cfg', 'yolov3.weights')

# Only COCO class 15 (cat)
CAT_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.5

# Setup screen capture
sct = mss.mss()
monitor = sct.monitors[1]

# Overlay window
class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(QtWidgets.QApplication.desktop().screenGeometry())
        self.detections = []  # list of (box, color)
        self.show()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        for (x, y, w, h, color) in self.detections:
            painter.setPen(QtGui.QPen(QtGui.QColor(*color), 3))
            painter.drawRect(x, y, w, h)

# Detect cats in one frame using a specific YOLO model
def detect_cats(net, out_layers, frame):
    h, w, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == CAT_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                center_x, center_y, width, height = detection[0:4] * np.array([w, h, w, h])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append((x, y, int(width), int(height)))
    return boxes

# Match boxes (simple IoU threshold)
def match_boxes(boxes1, boxes2, iou_threshold=0.5):
    matched = []
    unmatched1 = list(boxes1)
    unmatched2 = list(boxes2)

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    for box1 in boxes1:
        for box2 in boxes2:
            if iou(box1, box2) > iou_threshold:
                matched.append((box1, box2))
                unmatched1.remove(box1)
                unmatched2.remove(box2)
                break

    return matched, unmatched1, unmatched2

# Main app
app = QtWidgets.QApplication(sys.argv)
overlay = Overlay()

while True:
    start_time = time.time()

    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cats_v2 = detect_cats(net_v2, out_v2, frame)
    cats_v3 = detect_cats(net_v3, out_v3, frame)

    matched, only_v2, only_v3 = match_boxes(cats_v2, cats_v3)

    detections = []

    for (box2, box3) in matched:
        detections.append((box3[0], box3[1], box3[2], box3[3], (0, 255, 0)))  # green: both

    for box in only_v2:
        detections.append((box[0], box[1], box[2], box[3], (0, 0, 255)))  # blue: v2 only

    for box in only_v3:
        detections.append((box[0], box[1], box[2], box[3], (255, 0, 0)))  # red: v3 only

    overlay.detections = detections
    overlay.update()

    QtWidgets.QApplication.processEvents()

    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}", end="\r")
