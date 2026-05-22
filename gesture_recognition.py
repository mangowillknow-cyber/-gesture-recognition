import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
from collections import Counter


def count_extended_fingers(landmarks, hand_label):
    """Count extended fingers for one hand."""
    count = 0
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return count


def classify_gesture(hands_data):
    """Classify gesture from one or two hands."""
    if not hands_data:
        return "No hand detected"
    if len(hands_data) == 1:
        lm, label = hands_data[0]
        fingers = count_extended_fingers(lm, label)
        if fingers == 0:
            return "Fist"
        if fingers == 1:
            thumb_up = (lm[4].x < lm[3].x) if label == "Right" else (lm[4].x > lm[3].x)
            index_down = lm[8].y > lm[6].y
            if thumb_up and index_down:
                return "Thumbs Up"
            return "1"
        if fingers == 2:
            return "2"
        if fingers == 3:
            return "3"
        if fingers == 4:
            return "4"
        if fingers == 5:
            thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5
            if thumb_index_dist < 0.05:
                return "OK"
            return "5"
    total = sum(count_extended_fingers(lm, label) for lm, label in hands_data)
    if 1 <= total <= 10:
        return str(total)
    return "Unknown"


class CameraManager:
    def __init__(self):
        self.cap = None
        self.current_index = -1

    def enumerate_cameras(self, max_test=10):
        cameras = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        return cameras

    def open(self, index):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(index)
        self.current_index = index
        return self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label
                hands_data.append((hand_lm.landmark, label))
                self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return hands_data, frame

    def release(self):
        self.hands.close()


class GestureSmoother:
    def __init__(self, window=3):
        self.history = []
        self.window = window

    def update(self, gesture):
        self.history.append(gesture)
        if len(self.history) > self.window:
            self.history.pop(0)
        counts = Counter(self.history)
        return counts.most_common(1)[0][0]

    def reset(self):
        self.history.clear()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition")
        self.setMinimumSize(800, 600)

        self.camera = CameraManager()
        self.detector = HandDetector()
        self.smoother = GestureSmoother()

        self._setup_ui()
        self._init_camera()

        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(30)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top bar: camera selection
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_change)
        top.addWidget(self.camera_combo)
        self.status_label = QLabel("Status: Initializing...")
        top.addWidget(self.status_label)
        top.addStretch()
        layout.addLayout(top)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1a1a2e;")
        layout.addWidget(self.video_label)

        # Bottom: detected gesture
        self.gesture_label = QLabel("Detected: ---")
        self.gesture_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.gesture_label)

    def _init_camera(self):
        cameras = self.camera.enumerate_cameras()
        if not cameras:
            self.status_label.setText("Status: No camera found")
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)
        self._open_camera(cameras[0])

    def _on_camera_change(self, index):
        if index >= 0:
            cam_id = self.camera_combo.itemData(index)
            self._open_camera(cam_id)

    def _open_camera(self, cam_id):
        self.camera.release()
        self.smoother.reset()
        if self.camera.open(cam_id):
            self.status_label.setText(f"Status: Camera {cam_id} active")
        else:
            self.status_label.setText(f"Status: Failed to open Camera {cam_id}")

    def _process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        hands_data, frame = self.detector.detect(frame)
        gesture = classify_gesture(hands_data)
        smoothed = self.smoother.update(gesture)

        self.gesture_label.setText(f"Detected: {smoothed}")

        # Convert frame to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.timer.stop()
        self.detector.release()
        self.camera.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
