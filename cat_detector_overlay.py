import cv2
import numpy as np
import mss
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import pyautogui
import keyboard  # for hotkey
import time

KERNELSIZE = 608  # Size of the YOLO input image

# Load YOLOv2 and YOLOv3
def load_yolo(cfg, weights):
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

net_v2, out_v2 = load_yolo('yolov2.cfg', 'yolov2.weights')
net_v3, out_v3 = load_yolo('yolov3.cfg', 'yolov3.weights')

CAT_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.3

sct = mss.mss()
monitor = sct.monitors[1]
SCREEN_WIDTH = monitor["width"]
SCREEN_HEIGHT = monitor["height"]

# Mouse tracking square
def get_mouse_tracking_box():
    mouse_x, mouse_y = pyautogui.position()
    square_size = SCREEN_HEIGHT
    half_square = square_size // 2
    left = max(0, min(SCREEN_WIDTH - square_size, mouse_x - half_square))
    top = 0
    return {
        "left": left,
        "top": top,
        "width": square_size,
        "height": square_size
    }

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

def correct_boxes(boxes, scale, pad_x, pad_y, region_left, region_top, original_width, original_height):
    corrected = []
    for (x, y, w, h) in boxes:
        x = (x - pad_x) / scale
        y = (y - pad_y) / scale
        w = w / scale
        h = h / scale
        x = region_left + max(0, min(x, original_width - 1))
        y = region_top + max(0, min(y, original_height - 1))
        w = max(0, min(w, original_width - (x - region_left)))
        h = max(0, min(h, original_height - (y - region_top)))
        corrected.append((int(x), int(y), int(w), int(h)))
    return corrected

def detect_cats(net, out_layers, frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (KERNELSIZE, KERNELSIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes = []
    confidences = []
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
                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)
    final_boxes = [boxes[i] for i in indices]

    return final_boxes


class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(QtWidgets.QApplication.desktop().screenGeometry())
        self.detections = []  # list of (x, y, w, h, color)
        self.track_box = QtCore.QRect(0, 0, SCREEN_HEIGHT, SCREEN_HEIGHT)
        self.show()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 128))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(self.track_box, QtGui.QColor(0, 0, 0, 0))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        for (x, y, w, h, color) in self.detections:
            painter.setPen(QtGui.QPen(QtGui.QColor(*color), 3))
            painter.drawRect(x, y, w, h)

        # Add legend and counts at the bottom right of the track_box
        v2_count = sum(1 for _, _, _, _, color in self.detections if color == (255, 0, 0))
        v3_count = sum(1 for _, _, _, _, color in self.detections if color == (0, 255, 0))
        matched_count = sum(1 for _, _, _, _, color in self.detections if color == (255, 255, 0))

        legend_x = self.track_box.right() - 200
        legend_y = self.track_box.bottom() - 100

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 250)))
        painter.drawRect(legend_x, legend_y, 180, 80)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
        painter.drawText(legend_x + 10, legend_y + 60, f"Both YOLOv2 & YOLOv3: {matched_count}")

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
        painter.drawText(legend_x + 10, legend_y + 40, f"Just YOLOv3: {v3_count}")

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
        painter.drawText(legend_x + 10, legend_y + 20, f"Just YOLOv2: {v2_count}")

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
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    for box1 in boxes1:
        for box2 in boxes2:
            if iou(box1, box2) > iou_threshold:
                matched.append((box1, box2))
                unmatched1.remove(box1)
                unmatched2.remove(box2)
                break

    return matched, unmatched1, unmatched2

# Start Qt app
app = QtWidgets.QApplication(sys.argv)
overlay = Overlay()

print("Press F8 to take a snapshot!")

while True:
    region = get_mouse_tracking_box()
    overlay.track_box = QtCore.QRect(region["left"], region["top"], region["width"], region["height"])

    if keyboard.is_pressed('F8'):
        # Clear detections before capture
        overlay.detections = []
        overlay.update()
        QtWidgets.QApplication.processEvents()
        time.sleep(0.05)  # ensure update clears the overlay

        region_image = np.array(sct.grab(region))
        region_image = cv2.cvtColor(region_image, cv2.COLOR_BGRA2BGR)

        padded_frame, scale, pad_x, pad_y = resize_with_padding(region_image, (KERNELSIZE, KERNELSIZE))
        cats_v2 = detect_cats(net_v2, out_v2, padded_frame)
        cats_v3 = detect_cats(net_v3, out_v3, padded_frame)

        cats_v2 = correct_boxes(cats_v2, scale, pad_x, pad_y, region["left"], region["top"], region["width"], region["height"])
        cats_v3 = correct_boxes(cats_v3, scale, pad_x, pad_y, region["left"], region["top"], region["width"], region["height"])

        matched_boxes, cats_v2, cats_v3 = match_boxes(cats_v2, cats_v3)

        detections = []
        for box in cats_v2:
            detections.append((box[0], box[1], box[2], box[3], (255, 0, 0)))  # red = YOLOv2
        for box in cats_v3:
            detections.append((box[0], box[1], box[2], box[3], (0, 255, 0)))  # green = YOLOv3
        for box in matched_boxes:
            detections.append((box[0][0], box[0][1], box[0][2], box[0][3], (255, 255, 0))) # yellow = matched both

        overlay.detections = detections
        

        # Wait until F8 is released before allowing another snapshot
        while keyboard.is_pressed('F8'):
            time.sleep(0.1)
    overlay.update()

    QtWidgets.QApplication.processEvents()
