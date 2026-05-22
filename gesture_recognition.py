import os
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import sys
from collections import Counter


def count_extended_fingers(landmarks, hand_label):
    """Count extended fingers for one hand."""
    count = 0

    # Thumb: check x distance is significant enough
    thumb_tip_x = landmarks[4].x
    thumb_ip_x = landmarks[3].x
    thumb_distance = abs(thumb_tip_x - thumb_ip_x)
    if thumb_distance > 0.03:  # Minimum threshold for thumb extension
        if hand_label == "Right":
            if thumb_tip_x < thumb_ip_x:
                count += 1
        else:
            if thumb_tip_x > thumb_ip_x:
                count += 1

    # Other 4 fingers: check y distance is significant
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = landmarks[tip].y
        pip_y = landmarks[pip].y
        if pip_y - tip_y > 0.01:  # Minimum threshold for finger extension
            count += 1
    return count


def classify_gesture(hands_data):
    """Classify gesture from one or two hands."""
    if not hands_data:
        return "未检测到手势"
    if len(hands_data) == 1:
        lm, label = hands_data[0]
        fingers = count_extended_fingers(lm, label)
        if fingers == 0:
            return "握拳"
        if fingers == 1:
            thumb_up = (lm[4].x < lm[3].x) if label == "Right" else (lm[4].x > lm[3].x)
            index_down = lm[8].y > lm[6].y
            if thumb_up and index_down:
                return "点赞"
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
    return "未知手势"


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
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base_options

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
        base = mp_base_options.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_count = 0

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, self.frame_count)
        self.frame_count += 1

        hands_data = []
        if result.hand_landmarks:
            for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name
                hands_data.append((hand_lm, label))

        # Draw landmarks on frame
        if result.hand_landmarks:
            for hand_lm in result.hand_landmarks:
                for lm in hand_lm:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return hands_data, frame

    def release(self):
        self.landmarker.close()


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
        self.setWindowTitle("手势识别")
        self.setMinimumSize(800, 600)

        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon(icon_path))

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
        top.addWidget(QLabel("摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_change)
        top.addWidget(self.camera_combo)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_cameras)
        top.addWidget(self.refresh_btn)
        self.status_label = QLabel("状态: 初始化中...")
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
        self.gesture_label = QLabel("识别结果: ---")
        self.gesture_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.gesture_label)

    def _init_camera(self):
        cameras = self.camera.enumerate_cameras()
        if not cameras:
            self.status_label.setText("状态: 未检测到摄像头")
            return
        for idx in cameras:
            self.camera_combo.addItem(f"摄像头 {idx}", idx)
        self._open_camera(cameras[0])

    def _on_camera_change(self, index):
        if index >= 0:
            cam_id = self.camera_combo.itemData(index)
            self._open_camera(cam_id)

    def _refresh_cameras(self):
        self.camera.release()
        self.smoother.reset()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cameras = self.camera.enumerate_cameras()
        if not cameras:
            self.status_label.setText("状态: 未检测到摄像头")
        else:
            for idx in cameras:
                self.camera_combo.addItem(f"摄像头 {idx}", idx)
            self.camera_combo.blockSignals(False)
            self._open_camera(cameras[0])

    def _open_camera(self, cam_id):
        self.camera.release()
        self.smoother.reset()
        if self.camera.open(cam_id):
            self.status_label.setText(f"状态: 摄像头 {cam_id} 已连接")
        else:
            self.status_label.setText(f"状态: 无法打开摄像头 {cam_id}")

    def _process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        hands_data, frame = self.detector.detect(frame)
        gesture = classify_gesture(hands_data)
        smoothed = self.smoother.update(gesture)

        self.gesture_label.setText(f"识别结果: {smoothed}")

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
